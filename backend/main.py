"""ICP 故障诊断知识库后端入口。

保留通用业务路由与 Dify 外部知识库接口，产品界面收敛为七类文档管理、
ICP 故障诊断和系统设置。图片能力仅作为 FTA/RBD/架构证据保留，不再提供独立画廊。
"""
import os

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.storage.postgres_client import postgres_client
from backend.config import config
from backend.api import router as business_router
from backend.core.exceptions import BusinessError
from backend.core.response import fail

app = FastAPI(title="ICP Fault Diagnosis API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含文档、模型、检索、ICP 七类文档和 Dify 兼容接口。
app.include_router(business_router, prefix="/api/v1")

from backend.agent.api import router as agent_router  # noqa: E402
from backend.agent.db import agent_db  # noqa: E402

app.include_router(agent_router, prefix="/api")


@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(status_code=exc.code, content=fail(exc.code, exc.message, exc.detail))


@app.on_event("startup")
def _startup():
    """初始化 ICP 专用数据表、七类类型注册、默认诊断 Agent，并恢复中断任务。"""
    try:
        agent_db.init_tables()
    except Exception as exc:
        print(f"[Agent] 表初始化警告: {exc}")

    try:
        from backend.icp.repository import icp_repository
        icp_repository.init_tables()
    except Exception as exc:
        print(f"[ICP] 领域表初始化警告: {exc}")

    try:
        from backend.icp.document_types import normalize_document_types
        normalize_document_types()
    except Exception as exc:
        print(f"[ICP] 七类文档注册表规范化警告: {exc}")

    try:
        from backend.agent.icp_defaults import ensure_icp_diagnosis_agent
        ensure_icp_diagnosis_agent()
    except Exception as exc:
        print(f"[ICP] 默认诊断智能体初始化警告: {exc}")

    # 恢复因容器重启而卡住的文档状态。
    try:
        with postgres_client.conn.cursor() as cur:
            cur.execute("SELECT doc_id, file_name, status FROM documents WHERE status IN ('generating_kg', 'parsing')")
            stuck = cur.fetchall()
            for doc_id, file_name, status in stuck:
                new_status = "indexed" if status == "generating_kg" else "uploaded"
                cur.execute("UPDATE documents SET status=%s WHERE doc_id=%s", (new_status, doc_id))
                print(f"[Recovery] {file_name} ({doc_id[:8]}...) {status} -> {new_status}")
            if stuck:
                postgres_client.conn.commit()

            try:
                cur.execute("SELECT doc_id, file_name, status FROM icp_documents WHERE status IN ('generating_kg', 'parsing')")
                stuck_icp = cur.fetchall()
                for doc_id, file_name, status in stuck_icp:
                    new_status = "kg_pending" if status == "generating_kg" else "uploaded"
                    cur.execute("UPDATE icp_documents SET status=%s WHERE doc_id=%s", (new_status, doc_id))
                    print(f"[Recovery][ICP] {file_name} ({doc_id[:8]}...) {status} -> {new_status}")
                if stuck_icp:
                    postgres_client.conn.commit()
            except Exception as exc:
                print(f"[Recovery][ICP] 恢复失败: {exc}")
                try:
                    postgres_client.conn.rollback()
                except Exception:
                    pass

            import uuid as _uuid
            for table in ("kg_generation_logs", "document_parse_logs"):
                cur.execute(
                    f"""
                    SELECT DISTINCT task_id, doc_id, kb_id FROM {table}
                    WHERE task_id NOT IN (
                        SELECT task_id FROM {table}
                        WHERE log_type IN ('finish', 'cancel', 'error')
                    )
                    """
                )
                orphans = cur.fetchall()
                for task_id, doc_id, kb_id in orphans:
                    cur.execute(
                        f"""
                        INSERT INTO {table}
                            (log_id, task_id, doc_id, kb_id, log_level, log_type, message, progress, status)
                        VALUES (%s,%s,%s,%s,'INFO','cancel','[RECOVERY] 任务因服务重启被中断',0,'cancelled')
                        """,
                        (str(_uuid.uuid4()), task_id, doc_id, kb_id),
                    )
                if orphans:
                    postgres_client.conn.commit()
    except Exception as exc:
        print(f"[Recovery] 文档状态恢复失败: {exc}")
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass


static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 图片静态服务继续保留，供 FTA/RBD/系统架构原图和回答证据使用。
image_dir = config.IMAGE_STORAGE_DIR
os.makedirs(image_dir, exist_ok=True)
app.mount(config.IMAGE_URL_PREFIX, StaticFiles(directory=image_dir), name="images")

DIST_INDEX = os.path.join(static_dir, "dist", "index.html")


def _page_index() -> str:
    return DIST_INDEX


@app.get("/")
async def root():
    return FileResponse(_page_index())


@app.get("/diagnosis")
@app.get("/models")
@app.get("/agent")  # 旧书签兼容，由前端重定向到 /diagnosis。
async def spa_pages():
    return FileResponse(_page_index())


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(_page_index())
