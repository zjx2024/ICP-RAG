# ArtifactStore 本地文件存储方案

本方案将 ICP-RAG 的原始文档、工程图证据和解析产物从运行时 MinIO
服务迁移到 `ArtifactStore + LocalFS`。当前目标是单机可靠运行、完全本地
自部署和低运维复杂度，同时保留未来切换 S3 或分布式存储的业务边界。

## 1. 设计边界

- PostgreSQL 仍是权威业务事实来源。
- ArtifactStore 保存原始二进制文件和可回溯证据。
- OpenSearch 和 HugeGraph 仍是可重建的检索/图投影。
- 业务代码只使用逻辑 `storage_key`，不保存宿主机绝对路径。
- `/images` 只暴露工程图证据子目录，不恢复独立图片画廊。
- Dify 兼容路由仍能通过保留的逻辑 key 读取旧文档。

## 2. 运行架构

```text
上传 / 解析 / 证据读取
          │
          ▼
    ArtifactStore 接口
          │
          ▼
 LocalFSArtifactStore
          │
          └── /app/data/artifacts  <->  ./volumes/artifacts
```

当前后端由 `ARTIFACT_BACKEND=local` 选择 LocalFS。以后新增存储适配器时，
上传路由、七类解析器和 Dify 检索链路不应直接修改。

## 3. ArtifactStore 契约

核心操作：

| 操作 | 语义 |
|---|---|
| `upload_file` | 流式原子写入，返回 key、大小、SHA-256 和 MIME |
| `get_file` | 兼容小文件全量读取 |
| `open_file` | 只读流式访问 |
| `read_range` | 按 `start + length` 读取，超出 EOF 自动截断 |
| `materialize` | 为需要本地路径的解析库提供上下文路径 |
| `stat` / `exists` | 对象元数据和存在性检查 |
| `delete_file` | 幂等删除；本地后端默认移入可恢复 trash |

约定：

- 未找到对象时使用明确的 not-found 异常。
- 删除不存在对象返回 `False`。
- 真实 I/O 错误必须上抛，不允许记录“上传成功”后再静默丢失原件。
- `materialize` 的路径只在上下文中有效，业务表不得保存该路径。

## 4. 物理目录

LocalFS 使用 key-preserving 布局：

```text
volumes/artifacts/
├── .staging/                    # 同文件系统临时写入
├── .trash/                      # 延迟删除与误删恢复
├── icp/{doc_type}/{doc_id}/...  # 七类 ICP 原文
├── documents/{kb_id}/{doc_id}/... # 旧 Dify/知识库兼容原文
└── images/{doc_id}/...           # FTA/RBD/架构图证据
```

这一阶段不直接引入 `blobs/sha256/...` 去重。现有表没有统一的 artifact
引用计数，如果多个业务记录共享同一 blob，删除其中一个可能误伤另一个。
当第二阶段 `artifacts + artifact_refs` 完成后，才能安全转为内容寻址。

## 5. 原子性和安全

### 5.1 原子写入

```text
验证 storage_key
→ .staging 创建随机 .part
→ 分块写入并计算 size / SHA-256
→ flush + fsync
→ 同文件系统原子发布
→ fsync 父目录
```

`.staging` 和最终对象必须位于同一挂载点。容器中不得把 `.staging`
单独挂载到 `/tmp`。

### 5.2 key 安全

LocalFS 必须拒绝：

- 绝对路径、空 key、`.` 和 `..` 路径段；
- 反斜杠、NUL 字符和保留命名空间；
- 任何 resolve 后逃出 artifact root 的路径；
- 中间目录或目标文件为符号链接的情况。

用户上传的原文件名只是展示元数据。新 key 由 `doc_id` 和经白名单验证的扩展名生成，
不能把 `UploadFile.filename` 直接并入物理路径。

### 5.3 公开访问

FastAPI 的 `/images` 只指向：

```text
${ARTIFACT_STORAGE_DIR}/images
```

原始 Word/Excel、解析 JSON 和 trash 均不允许通过静态路由下载。如需原文下载，
应由应用 API 完成权限检查后流式返回。

## 6. PostgreSQL 兼容策略

第一阶段保留以下物理列：

```text
documents.minio_path
icp_documents.minio_path
document_images.minio_path
```

这些列的值解释为中性 `storage_key`。由于迁移前后 key 完全不变，无需
对生产数据做破坏性 rename/update。新 API 可增加 `storage_key` 字段，但需在一个
兼容周期内继续返回 `minio_path`。

目标数据模型：

```text
artifacts
  artifact_id / backend / storage_key / sha256 / size_bytes
  content_type / original_name / status / timestamps

artifact_refs
  artifact_id / owner_kind / owner_id / doc_id / role / timestamps
```

ICP 文档使用独立 `icp_documents`，而存量图片/图结构的部分外键只面向旧
`documents`。因此第二阶段应使用可表达 `owner_kind + owner_id` 的独立引用表，
不应仅给某一个旧表增加外键。

## 7. PostgreSQL 与 LocalFS 一致性

两者无法参与同一 ACID 事务。当前安全顺序为：

```text
原子写入 artifact
→ PostgreSQL 登记文档
→ PG 报错时 rollback 后按 doc_id 读回确认
→ 仅在确认记录不存在时将 artifact 移入 trash
```

提交响应中断时可能出现“服务端已提交、客户端收到异常”。此时读回到记录就按
成功登记处理；如果连读回也失败，则保留原件并报“提交结果待审计”，不能冒险
补偿删除。该策略最多留下可清理孤儿，不会主动制造指向缺失原件的 PG 记录。
定期一致性任务应检查：

- 超过阈值的 `.staging` 文件；
- PG 引用但 LocalFS 缺失的对象；
- LocalFS 存在但 PG 无引用的孤儿对象；
- trash 保留期和可恢复性；
- 重点原件的 SHA-256 变化。

备份时建议先获取 PostgreSQL 一致快照，再快照 `volumes/artifacts`。Artifact
写入在 PG 登记之前完成，且删除会进入 trash，因此这个顺序不会让 PG 快照
指向一个无法恢复的对象。

### 7.1 只读一致性审计

`scripts/audit_artifact_refs.py` 用于部署验收和定期巡检。它扫描：

```text
documents.minio_path
icp_documents.minio_path
document_images.minio_path
```

审计在 PostgreSQL 只读事务中执行，只对现有 LocalFS 目录做 `stat`/遍历；不会
创建 ArtifactStore 目录、修改数据库、移动孤儿或清理文件。默认只校验数据库
引用是否指向普通文件：

```bash
POSTGRES_HOST=localhost \
POSTGRES_PORT=5433 \
POSTGRES_USER=postgres \
POSTGRES_PASSWORD='replace-me' \
POSTGRES_DB=knowledge_graph \
python scripts/audit_artifact_refs.py \
  --artifact-root ./volumes/artifacts
```

同时清点未被上述三列引用的 LocalFS 孤儿，并排除 `.staging`、`.trash`：

```bash
python scripts/audit_artifact_refs.py \
  --artifact-root ./volumes/artifacts \
  --check-orphans
```

`--report-orphans` 是 `--check-orphans` 的别名；`--json` 可输出机器可读结果。
退出码可直接用于部署流水线：

| 退出码 | 含义 |
|---:|---|
| `0` | 本次选择的检查项一致 |
| `1` | 存在缺失/非法引用，或启用孤儿检查后发现孤儿 |
| `2` | 数据库不可用、挂载目录缺失或审计无法完整执行 |

孤儿只做报告，不能据此自动删除。旧图片迁移后可能存在尚未登记到
`document_images` 的证据文件，应先人工确认来源和保留策略。

## 8. MinIO 迁移

### 8.1 为什么不能直接拷贝卷

`volumes/minio` 中的 `xl.meta`、`.minio.sys` 和分片文件是 MinIO 内部格式，
小对象数据还可能内联在 `xl.meta` 中。必须使用 MinIO API 读取对象流，
不能对数据卷执行普通文件拷贝作为迁移。

### 8.2 安装一次性依赖

```bash
python -m venv .venv-migration
source .venv-migration/bin/activate
pip install -r requirements-migration.txt
```

MinIO SDK 不在主 `requirements.txt` 中，不会进入运行时应用镜像。

### 8.3 建议窗口

1. 停止 app 及所有上传/解析后台任务。
2. 记录旧 app、Compose 和 MinIO 的精确镜像 ID，备份 PostgreSQL、
   `volumes/minio` 和 `app/storage/images`。
3. 先复制 MinIO 数据卷，仅在可写副本上用同一个旧镜像 ID 启动迁移源。
4. 先 dry-run，再迁移，最后 verify-only。
5. 切换应用后完成上传、原文、解析、图片和删除冒烟。
6. 历史 MinIO 卷和 manifest 至少只读保留 7–30 天。

不要用一个未经验证的新 MinIO 版本直接打开唯一的旧卷，它可能修改
元数据格式。

记录并启动可复现的迁移源（示例容器名按实际调整）：

```bash
# 必须在删除旧容器前记录不可变 image ID，不能把 latest 当版本锁。
LEGACY_MINIO_IMAGE="$(docker inspect --format '{{.Image}}' icp-minio-1)"
docker image inspect "$LEGACY_MINIO_IMAGE" >/dev/null

# 先停止旧 MinIO，再制作卷副本；下面的副本允许旧服务写自身元数据，原卷不挂载。
sudo cp -a volumes/minio volumes/minio-migration-copy

docker run -d --rm --name icp-minio-migration-source \
  -p 127.0.0.1:9002:9000 -p 127.0.0.1:9003:9001 \
  -e MINIO_ROOT_USER="$MIGRATION_MINIO_ACCESS_KEY" \
  -e MINIO_ROOT_PASSWORD="$MIGRATION_MINIO_SECRET_KEY" \
  -v "$PWD/volumes/minio-migration-copy:/data" \
  "$LEGACY_MINIO_IMAGE" server /data --console-address ':9001'
```

若旧容器已删除且没有保存 image ID/digest，应先从备份恢复原镜像；不要拉取
`minio/minio:latest` 猜测版本。

### 8.4 迁移命令

```bash
# 只列出工作，不读取对象内容
python scripts/migrate_minio_to_localfs.py --dry-run

# 迁移 MinIO documents bucket，并同时迁移旧 app/storage/images
MIGRATION_MINIO_ENDPOINT=localhost:9002 \
MIGRATION_MINIO_ACCESS_KEY=admin \
MIGRATION_MINIO_SECRET_KEY='replace-me' \
python scripts/migrate_minio_to_localfs.py --resume

# 流式重读源与目标，比较 size + SHA-256
python scripts/migrate_minio_to_localfs.py --verify-only

# 当 MinIO 无对象、仅需迁移旧本地图片时
python scripts/migrate_minio_to_localfs.py --skip-minio --resume
```

默认 manifest：

```text
volumes/migration/minio-to-localfs.jsonl
```

每条记录包含 source、bucket、key、ETag/本地时间标记、预期大小、SHA-256、
状态和错误。`--resume` 只会跳过 manifest 中已成功且目标 SHA-256 仍一致的对象。
如果目标 key 已存在但内容不同，迁移会报错并保留两端原状，不覆盖。

## 9. Docker 配置

运行时环境变量：

```env
ARTIFACT_BACKEND=local
ARTIFACT_STORAGE_DIR=/app/data/artifacts
ARTIFACT_TRASH_ENABLED=true
IMAGE_STORAGE_DIR=/app/data/artifacts/images
```

宿主机挂载：

```yaml
volumes:
  - ./volumes/artifacts:/app/data/artifacts
```

主 Compose 不再包含 MinIO 服务、端口、健康检查或 `depends_on`。旧
`volumes/minio` 是受保护的迁移源，移除运行时服务不等于授权删除旧卷。

## 10. 验收与回切

切换前必须满足：

- manifest 无 `error`；
- `--verify-only` 对全部源对象返回成功；
- PostgreSQL 中非空 storage key 在 LocalFS 的存在率为 100%；
- 旧工程图 URL 及新解析图片均可访问；
- 七类文档上传/原文/解析/删除通过；
- Dify 两个兼容路由仍正常挂载；
- 容器重建后 artifact 和 `/images` 仍存在。

### 10.1 前向修复

如切换后发现少量缺失，优先前向修复：

1. 立即停止 app 写入。
2. 保留当前 `volumes/artifacts` 作为调查证据。
3. 使用 manifest 定位缺失/冲突 key，从保留的 MinIO 卷副本重跑迁移。
4. 在未验证完成前，不删除历史 MinIO 卷或一次性迁移镜像。

### 10.2 可执行回切窗口

真正回切必须在“切换后仍禁止上传”的冒烟窗口内完成。切换前需保留：旧 app
镜像、旧 Compose、精确 MinIO 镜像 ID、MinIO 卷副本和同一时刻的 PostgreSQL
快照。回切步骤：

1. 停止新 app，复制保留当前 `volumes/artifacts` 和日志作为调查证据。
2. 停止新栈；用切换前 PostgreSQL 快照恢复到独立恢复目录/实例。
3. 用旧 Compose、旧 app 镜像、精确旧 MinIO 镜像和旧卷副本启动旧栈。
4. 对文档数量、原文读取、图片和 Dify 检索做只读验收，确认后再开放写入。
5. 新 ArtifactStore 与 migration manifest 保留，不能在回切过程中删除。

一旦 LocalFS 新栈已经接受新上传或业务修改，直接恢复旧 PG 快照会丢新数据，旧
app 也无法读取这些 LocalFS 新对象，因此不再允许“一键回切”。此时必须保持维护
模式，逐条把新增 PG 记录及 artifact 按原 key 对账回灌后才能回切；通常应选择
前向修复。该边界也是建议保留 7–30 天旧卷、但只在短暂零写窗口承诺回切的原因。

## 11. 适用性边界

LocalFS 适合当前单机或同一共享挂载上的应用实例。当出现以下需求时，
应通过 ArtifactStore 新增对象存储适配器：

- 多台物理服务器需要同时读写；
- 需要跨节点副本和自动故障转移；
- 需要 S3 API、预签名 URL 或服务端生命周期策略；
- 数据规模和并发量已经超过单机磁盘边界。

只需增加新适配器和数据迁移，不应让解析器、上传 API 或诊断工具感知具体存储产品。
