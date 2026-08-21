"""后端新入口：backend.main:app。

与 app.main:app 行为完全一致（路由、startup 恢复、静态文件、页面路由均复用），
仅入口包名变化。存量 app 包保留，可随时回滚。
"""
import json
import os
from urllib.parse import urlparse

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.storage.postgres_client import postgres_client

from backend.config import config
from backend.api import router as business_router
from backend.core.exceptions import BusinessError
from backend.core.response import fail

app = FastAPI(title="Knowledge Graph API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(business_router, prefix="/api/v1")

# ===== Agent 模块（新增，零侵入：仅新增路由，现有路由不动）=====
from backend.agent.api import router as agent_router  # noqa: E402
from backend.agent.db import agent_db  # noqa: E402

app.include_router(agent_router, prefix="/api")


@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(status_code=exc.code, content=fail(exc.code, exc.message, exc.detail))


@app.on_event("startup")
def _init_agent_tables():
    """启动时初始化 Agent 专属表（新建表，不动现有表）"""
    try:
        agent_db.init_tables()
    except Exception as e:
        print(f"[Agent] 表初始化警告: {e}")

    # 恢复因容器重启而卡住的文档状态
    # generating_kg/parsing 是中间态，容器重启后后台任务已丢失，需回退到前一个稳定态
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT doc_id, file_name, status FROM documents WHERE status IN ('generating_kg', 'parsing')")
            stuck = cur.fetchall()
            for doc_id, file_name, status in stuck:
                new_status = 'indexed' if status == 'generating_kg' else 'uploaded'
                cur.execute("UPDATE documents SET status = %s WHERE doc_id = %s", (new_status, doc_id))
                print(f"[Recovery] {file_name} ({doc_id[:8]}...) {status} -> {new_status}")
            if stuck:
                postgres_client.conn.commit()
                print(f"[Recovery] 恢复了 {len(stuck)} 个卡住的文档")

            # ICP 类型驱动文档同样恢复中间态（parsing -> uploaded, generating_kg -> kg_pending）
            try:
                cur.execute("SELECT doc_id, file_name, status FROM icp_documents WHERE status IN ('generating_kg', 'parsing')")
                stuck_icp = cur.fetchall()
                for doc_id, file_name, status in stuck_icp:
                    new_status = 'kg_pending' if status == 'generating_kg' else 'uploaded'
                    cur.execute("UPDATE icp_documents SET status = %s WHERE doc_id = %s", (new_status, doc_id))
                    print(f"[Recovery][ICP] {file_name} ({doc_id[:8]}...) {status} -> {new_status}")
                if stuck_icp:
                    postgres_client.conn.commit()
                    print(f"[Recovery][ICP] 恢复了 {len(stuck_icp)} 个卡住的文档")
            except Exception as e:
                print(f"[Recovery][ICP] 恢复失败: {e}")
                try:
                    postgres_client.conn.rollback()
                except Exception:
                    pass

            # 标记因重启而中断的孤立任务（有日志但无终止日志的任务）
            import uuid as _uuid
            for table in ('kg_generation_logs', 'document_parse_logs'):
                cur.execute(f"""
                    SELECT DISTINCT task_id, doc_id, kb_id FROM {table}
                    WHERE task_id NOT IN (
                        SELECT task_id FROM {table}
                        WHERE log_type IN ('finish', 'cancel', 'error')
                    )
                """)
                orphans = cur.fetchall()
                for task_id, doc_id, kb_id in orphans:
                    cur.execute(f"""
                        INSERT INTO {table} (log_id, task_id, doc_id, kb_id, log_level, log_type, message, progress, status)
                        VALUES (%s, %s, %s, %s, 'INFO', 'cancel', '[RECOVERY] 任务因服务重启被中断', 0, 'cancelled')
                    """, (str(_uuid.uuid4()), task_id, doc_id, kb_id))
                    print(f"[Recovery] 标记孤立任务为已取消: {table} task={task_id[:8]}...")
                if orphans:
                    postgres_client.conn.commit()
                    print(f"[Recovery] 标记了 {len(orphans)} 个 {table} 中的孤立任务")
    except Exception as e:
        print(f"[Recovery] 文档状态恢复失败: {e}")
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

image_dir = config.IMAGE_STORAGE_DIR
os.makedirs(image_dir, exist_ok=True)
app.mount(config.IMAGE_URL_PREFIX, StaticFiles(directory=image_dir), name="images")

# ===== 页面路由（新 React 前端，产物在 static/dist，唯一前端入口） =====
# 旧版单页（static/ 根目录的 index.html/agent.html/gallery.html 等）已随重构移除，
# 全部页面统一由 React SPA 提供。
DIST_INDEX = os.path.join(static_dir, "dist", "index.html")


def _page_index() -> str:
    return DIST_INDEX


@app.get("/")
async def root():
    return FileResponse(_page_index())


@app.get("/upload")
@app.get("/retrieval")
@app.get("/view")
@app.get("/visualize")
@app.get("/models")
@app.get("/logs")
@app.get("/graph")
@app.get("/gallery")
@app.get("/agent")
async def spa_pages():
    """前端路由（BrowserRouter）统一返回 SPA 入口。"""
    return FileResponse(_page_index())


@app.get("/api/v1/images")
async def list_images(kb_id: str = None):
    """列出所有有图片的文档（从 PostgreSQL 查询）；传入 kb_id 时只统计该知识库。"""
    try:
        docs = postgres_client.get_all_image_docs(kb_id)
        return {"documents": docs, "total_documents": len(docs)}
    except Exception:
        return {"documents": [], "total_documents": 0}


@app.get("/api/v1/images/{doc_id}")
async def list_images_by_doc(doc_id: str):
    """列出指定文档的所有图片（含 LLM 解析内容）和表格。"""
    images_with_desc = []
    tables_with_desc = []

    try:
        db_images = postgres_client.get_images_by_doc(doc_id)
        for img in db_images:
            image_url = img.get("image_url", "")
            rel_url = ""
            if image_url:
                if image_url.startswith("http"):
                    parsed = urlparse(image_url)
                    rel_url = parsed.path
                else:
                    rel_url = image_url
            if not rel_url:
                rel_url = f"/images/{doc_id}/placeholder"

            images_with_desc.append({
                "image_id": img.get("image_id", ""),
                "title": img.get("title", ""),
                "description": img.get("description", ""),
                "image_type": img.get("image_type", "normal"),
                "image_url": rel_url,
                "chunk_id": img.get("chunk_id", ""),
                "minio_path": img.get("minio_path", ""),
                "created_at": str(img.get("created_at", "")) if img.get("created_at") else None
            })
    except Exception:
        pass

    try:
        db_tables = postgres_client.get_tables_by_doc(doc_id)
        for tbl in db_tables:
            meta = tbl.get("meta_data", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            elif meta is None:
                meta = {}

            tables_with_desc.append({
                "table_id": tbl.get("table_id", ""),
                "title": tbl.get("title", ""),
                "description": tbl.get("description", ""),
                "meta_data": meta,
                "chunk_id": tbl.get("chunk_id", ""),
                "created_at": str(tbl.get("created_at", "")) if tbl.get("created_at") else None
            })
    except Exception:
        pass

    return {
        "doc_id": doc_id,
        "images": images_with_desc,
        "tables": tables_with_desc,
        "image_count": len(images_with_desc),
        "table_count": len(tables_with_desc)
    }


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """SPA 兜底：未匹配的路径返回前端入口（避免直接刷新子路由 404）。

    API 路径（api 开头）不返回 SPA 入口：无斜杠的 API 请求交由 FastAPI 的
    redirect_slashes 处理（307 → 带斜杠路由），避免被本兜底路由吞掉返回 HTML。
    """
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(_page_index())
