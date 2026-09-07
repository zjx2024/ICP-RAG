# ICP-RAG 修缮进度复审

> 复审日期：2026-09-04
> 分支：`refactor/icp-diagnosis-specialization`
> 对照文档：[ICP工程质量修缮路线图.md](./ICP工程质量修缮路线图.md)
> 核实基线：提交 `fe1acaa`（B2 评测闭环）起往前 20 个提交

本文档记录路线图落地一轮后的复审结论。**所有 `文件:行号` 均经实际命令核实**，核实方法见 [附录 A](#附录-a核实命令)。

---

## 1. 结论摘要

**执行力度很强，但拆分停留在"物理搬家"层面，且新增代码缺少测试护栏。**

看得见的债还得干净（CI、安全基线、死代码、构建产物、工作区），看不见的债基本没动（依赖方向、巨型函数、测试护栏）。

| 维度 | 复审前 | 本次 | 变化 | 依据 |
|---|:---:|:---:|:---:|---|
| 领域设计与安全性 | 8.5 | 8.5 | — | 本轮未触及，资产完好 |
| 测试有效性 | 7.5 | **7.0** | ↓ | 搬迁约 7,000 行仅增 41 用例；P0-A 修复零测试 |
| 工程规范与可维护性 | 4 | **5.5** | ↑ | 拆分启动、CI 修复、死代码清理；但循环依赖未解 |
| 前端质量 | 5 | **5.5** | ↑ | 页面/hook/封装已清；`any` 57 处、无 ESLint 未变 |
| 交付就绪度 | 5 | **8.5** | ↑↑ | 工作区 291 → **0**，CI 已修，安全基线收紧 |
| **综合** | **5.5** | **7.0** | ↑ | |

### 1.1 体量变化

| 指标 | 复审前 | 本次 | 变化 |
|---|---:|---:|---:|
| 后端 Python | 74,090 | **75,612** | +1,522 |
| `answer.py` | 14,644 | **7,993** | −6,651 |
| 新增域模块 | — | 7,374 | 4 个域 + `answer_presence` |
| 前端 TS/TSX | 9,073 | 9,005 | −68 |
| 测试文件 | 74 | **78** | +4 |
| 测试用例 | 949 | **990** | +41 |
| 未提交改动 | 291 | **0** | 全部落地 |

---

## 2. 已完成且已验证

| 路线图项 | 验证命令结果 |
|---|---|
| 步骤 0 落地工作区 | `git status --porcelain \| wc -l` = **0** |
| P0-1 CI `pip install` | `.github/workflows/ci.yml:27` 已补 |
| P0-2 `HugeGraphUnavailable` | 已引入，区分空结果与服务故障 |
| P0-3 图谱 `print` | `grep -c 'print(' hugegraph_client.py` = **0**（原 97） |
| P0-5 死解析器注册 | `text.py` / `fault_tree.py` 已删，覆盖隐患消除 |
| P0-6 安全基线 | CORS 白名单、容器非 root、口令可强制 |
| P0-7 `call_model` 收敛 | 私有调用 **16 → 0** |
| P1-3 `MockPostgres` 显式开关 | `conn_is_mock` 替代 `.__class__.__name__` 字符串判断 |
| P1-5 SQL 标识符白名单 | 已补 |
| P1-6 迭代上限统一 | 已统一配置源 |
| P1-7 `static/dist` 移出版本控制 | `git ls-files static/dist \| wc -l` = **0**（原 66） |
| P1-10 前端死代码 | `TypeLogs.tsx`、`hooks/` 三个 SWR hook、`kb.ts:32` 已清 |
| B1 全链路 tracing | `agent/tracing.py`(290) + `api/tracing_router.py`(30)，接入 `chat_router.py:21` |
| B2 评测门槛 | `_gate_threshold` @ `scripts/evaluate_icp_paraphrases.py:1581` |
| 回归测试 | **990 全绿（21.7s）** |
| 后端编译 | 通过 |

> **值得肯定**：提交历史高度聚焦，每批对应一个路线图条目（`C1 批 1..6`、`P0-2`、`P0-5`……），工作区保持清洁。这是可审查、可回滚的正确推进方式。

---

## 3. 核心问题：拆分的三个结构缺陷

`answer.py` 由 14,644 行降至 **7,993 行**，迁出四个域模块。数字漂亮，但**依赖图没有真正解开**。

### 3.1 双向依赖，不是解耦

四个域模块全部在文件头部反向依赖 `answer.py`：

```
answer_domain_calculation.py:22  ─┐
answer_domain_fault.py:18        ─┤ from backend.agent.nodes.answer import (...)
answer_domain_history.py:24      ─┤   合计 75 个符号
answer_domain_target.py:21       ─┘   （target 29 / fault 19 / history 16 / calculation 11）
```

而 `answer.py` 在**文件末尾**反向导入四个域模块：

```python
7832: from backend.agent.nodes.answer_domain_fault import (  # noqa: E402
7873: from backend.agent.nodes.answer_domain_history import (  # noqa: E402
7908: from backend.agent.nodes.answer_domain_target import (  # noqa: E402
7972: from backend.agent.nodes.answer_domain_calculation import (  # noqa: E402
```

**这是靠"把 import 延迟到文件末尾"打破循环，靠 `# noqa: E402` 抑制告警。** 已验证当前可正常 import（`import backend.agent.nodes.answer` 成功，导出 395 个符号），但只要有人在 `answer.py` 顶部加一行域模块 import，就会立即崩溃。

这不是解耦，是**物理搬家**。

### 3.2 总量不降反增

```
拆分前   answer.py                     14,644
拆分后   answer.py          7,993
       + domain_fault       2,794
       + domain_history     1,378
       + domain_target      2,243
       + domain_calculation   959
       + answer_presence     105   ───────── 15,472   (+828)
```

搬迁产生了约 800 行适配与重复开销。

### 3.3 巨型函数基本未解决

`answer.py` 仍剩 **129 个顶层函数**，最大五个：

| 行数 | 函数 | 位置 |
|---:|---|---|
| 442 | `answer_node` | `:7385` |
| 354 | `_filter_answer_domain_evidence` | `:1304` |
| 308 | `_select_evidence_window` | `:640` |
| 298 | `_target_completion_manifest` | `:4632` |
| **279** | **`_target_facet_is_present`** | `:4351` |

**特别注意最后一行**：提交 `4c4f68a`（C1 批 1）的描述是「抽取 `answer_presence` 共享字段在场原语」，但 `_target_facet_is_present` 这个 279 行函数**仍留在 `answer.py`**，没有迁走。批 1 的目标并未真正达成。

**域间耦合同样未切断**：`answer_domain_history.py:23` 导入了 `answer_domain_fault._prompt_payload_as_evidence` —— **下划线开头的私有函数被跨模块引用**。

---

## 4. 未完成项

| 路线图项 | 状态 | 证据 |
|---|---|---|
| P0-4 全局静默异常 | **60 → 61，恶化** | `grep -c 'except Exception: … pass'` = 61；仅修了"回滚失败"一处 |
| C2 护栏单一来源 | 未动 | 4 份别名表原封不动：`adaptive_tools.py:106`、`scope_router.py:46`/`:58`、`answer_contract.py:256`/`:308`、`tool_policy.py:38` |
| C3 结构化输出契约化 | 未动 | `response_format` 在 plan / adaptive_tools / scope_router 中 **0 处** |
| C4 AgentState 治理 | 未动 | `diagnostic_confidence` 仍有 **9 处** |
| 排序权重外置 | 未动 | 仍硬编码于 `icp_diagnosis.py:3483` `0.55 / 0.45 / 0.3` |
| 前端质量门 | 未动 | `any` 仍 **57 处**；无 ESLint；`swr` 依赖未卸载（`package.json:21`） |
| P2 全部 | 未动 | 版本化、复核闭环、检索精修、延迟优化 |

> 补充：`approved_ids` 构造点由 **13 处收敛到 2 处**，`approved_id_set` 达 18 处 —— **C1 批 2 是有效的**，这一项做对了。

---

## 5. 最需要警惕的两件事

### 5.1 P0-A 的修复没有任何测试护栏

```
grep -rl "HugeGraphUnavailable" tests/   →  0 个文件
grep -rl "conn_is_mock"         tests/   →  0 个文件
ls tests/ | grep -i trace                →  0 个文件
```

含义是：CI 补的 `pip install`、引入的 `HugeGraphUnavailable`、`conn_is_mock` 开关、SQL 白名单、290 行的 `tracing.py`——**全部可以被后续提交无声回退，而 CI 依然绿灯**。

修复了但没守住，等于修了一半。而 `tracing.py` 是后续所有度量（B1/B2）的地基，零测试意味着这个地基本身不可信。

### 5.2 100 例多轮评测集从未被接线

三份文件确实存在于 `evals/`：

```
icp_multiturn_regression_positioning.json   (33 例)
icp_multiturn_regression_analysis.json      (35 例)
icp_multiturn_regression_boundary.json      (32 例)
```

但 `grep -rn 'icp_multiturn_regression'` 全仓零引用，`.github/workflows/ci.yml` 也不跑评测。

**B2 造好了"评测门槛"参数，却没有评测集被喂进去——门槛是空的。**

这是当前最讽刺的一点：仪表盘造好了、传感器修好了，但没接上被测对象。

---

## 6. 建议的下一步

| 优先级 | 动作 | 工作量 | 理由 |
|---|---|---|---|
| **1** | 给 P0-A 四项修复 + `tracing.py` 补测试 | 1 天 | 否则随时可无声回退，前面的功夫白费 |
| **2** | 把 100 例评测集接进 CI（需先解决真实服务依赖：PG / OpenSearch / HugeGraph / LLM） | 1–2 天 | 让 B1 / B2 真正产出数据，门槛才有意义 |
| **3** | **切断循环依赖**：把 75 个反向依赖符号抽到 `answer_pipeline.py` 中立模块，使 `answer.py` 与四个域模块**只依赖它、互不依赖** | 2–3 天 | 现在的拆分在依赖图上是假的 |
| **4** | 继续拆 `answer.py` 剩余 7,993 行，重点将 `_target_facet_is_present`(279 行, `:4351`) 真正迁入 `answer_presence` | 随 3 一起做 | C1 批 1 的收尾 |
| **5** | C2 别名表 4→1、C3 结构化输出、C4 清 `diagnostic_confidence`、权重外置 | 3–4 天 | P1 剩余项 |

### 6.1 关于第 3 项的具体做法

现状是 `answer.py` 与四个域模块互相 import。建议引入中立模块承载共享原语，使依赖变为单向：

```
现状（双向）：
    answer.py  ┄┄┄▶  answer_domain_*.py
         ◀┄┄┄┄┄┄┄┄┄┄┘   （延迟到底部 import + noqa: E402）

目标（单向）：
    answer.py           ──▶  answer_pipeline.py
    answer_domain_*.py  ──▶  answer_pipeline.py   （共享原语、状态读取、证据工具）
    answer.py           ──▶  answer_domain_*.py   （仅顶层调用，可放文件头部）
```

75 个反向依赖的符号应按性质分流：纯函数（格式化、判定、清洗）进 `answer_pipeline.py`；需要节点状态的读取函数改为显式传参而非闭包捕获。

---

## 7. 保持不动的部分（复核确认）

以下资产在本轮未被触及，状态完好，**继续禁止回退**：

- `icp_calculation.py` 纯确定性计算，无 LLM 参与
- `reliability.py` 共享基本事件 fail-closed、路径超限拒绝出数值
- `graph.py:125-139` 图关系重授权 fail-closed
- `diagram_validation.py` 视觉结果最高只到 `provisional`，低置信隔离
- 精确 HMC 优先匹配 PostgreSQL，语义命中须回 PG 验真
- 无自主反思循环、无长期记忆、无第八类知识卡片

---

## 附录 A：核实命令

本文档所有数据均可用以下命令复现（与路线图文档附录 A 同一纪律：**引用必附可复现命令**）。

```bash
# 体量
find backend -name '*.py' | xargs wc -l | sort -rn | sed -n '2,16p'
find frontend/src -name '*.ts' -o -name '*.tsx' | xargs wc -l | tail -1

# 测试
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
grep -rc 'def test_' tests/*.py | awk -F: '{s+=$2} END {print s}'

# 循环依赖（双向 import 点）
grep -n '^from backend.agent.nodes.answer import' backend/agent/nodes/answer_domain_*.py
grep -n '^from backend.agent.nodes.answer_domain' backend/agent/nodes/answer.py

# 巨型函数（按 AST 统计，比 grep 准确）
.venv/bin/python - <<'PY'
import ast
t = ast.parse(open('backend/agent/nodes/answer.py', encoding='utf-8').read())
fs = [(n.end_lineno - n.lineno + 1, n.name, n.lineno)
      for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
for l, name, ln in sorted(fs, reverse=True)[:8]:
    print(f'{l:5d}  {name}  (:{ln})')
PY

# 剩余债务计数
grep -rn -A1 'except Exception:' backend --include=*.py | grep -c '^\S*[-:] *pass$'
grep -rn 'diagnostic_confidence' backend --include=*.py | wc -l
grep -rn 'response_format' backend/agent/nodes/plan.py backend/agent/nodes/adaptive_tools.py backend/agent/nodes/scope_router.py | wc -l
grep -rn ': any\|as any\|<any>' frontend/src --include=*.ts --include=*.tsx | wc -l

# 测试护栏缺失确认
grep -rl 'HugeGraphUnavailable' tests/ | wc -l
grep -rl 'conn_is_mock' tests/ | wc -l
ls tests/ | grep -i trace | wc -l

# 评测集接线确认
grep -rn 'icp_multiturn_regression' --include=*.py --include=*.yml --include=*.sh . | grep -v node_modules | wc -l
```

### 未能验证的项

- **前端构建**：本机无 `npm`，`npm ci && npm run build` 未实际执行
- **真实服务联调**：PG / OpenSearch / HugeGraph / LLM 不可用，100 例评测集**从未实跑**
- **CI 真实运行结果**：GitHub Actions 执行记录未获取
