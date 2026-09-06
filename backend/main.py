"""ICP 故障诊断知识库后端入口。

保留通用业务路由与 Dify 外部知识库接口，产品界面收敛为七类文档管理、
ICP 故障诊断和系统设置。图片能力仅作为 FTA/RBD/架构证据保留，不再提供独立画廊。
"""
import os
import psycopg2

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.storage.postgres_client import postgres_client
from backend.storage.search_store import search_store
from backend.config import config
from backend.api import router as business_router
from backend.core.exceptions import BusinessError
from backend.core.response import fail

# 生产强口令检查：开启强口令开关后，STRICT 模式不得以空口令或旧默认值
# "password" 启动，强制部署方显式注入真实凭据（见 config.py）。
if (
    config.STRICT_DEPENDENCIES
    and config.ICP_REQUIRE_STRONG_POSTGRES_PASSWORD
    and (not config.POSTGRES_PASSWORD or config.POSTGRES_PASSWORD == "password")
):
    raise RuntimeError(
        "ICP_REQUIRE_STRONG_POSTGRES_PASSWORD=true 且 STRICT_DEPENDENCIES=true 时，"
        "必须通过环境变量 POSTGRES_PASSWORD 显式提供非默认口令"
    )

app = FastAPI(title="ICP Fault Diagnosis API", version="3.0.0")

# CORS 白名单：默认只放行本机/局域网开发入口，去掉通配符 "*"。
# 允许跨域访问的来源需通过 CORS_ALLOW_ORIGINS（逗号分隔）显式配置，
# 例如云端网关域名或内网可视化端。allow_credentials=True 不能与 "*" 并用，
# 浏览器会拒绝携带凭据的跨域请求，这里改为明确的本机来源列表。
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5174",   # Vite 开发服务器
    "http://127.0.0.1:5174",
    "http://localhost:19999",  # 本地热启动
    "http://127.0.0.1:19999",
    "http://localhost:9999",   # Docker 容器自访问（健康检查/探活）
    "http://127.0.0.1:9999",
]
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
] or _DEFAULT_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含文档、模型、检索、ICP 七类文档和 Dify 兼容接口。
app.include_router(business_router, prefix="/api/v1")

from backend.agent.api import router as agent_router  # noqa: E402
from backend.agent.db import agent_db  # noqa: E402

app.include_router(agent_router, prefix="/api")


def _dependency_status():
    """Return non-secret readiness details for the four local data services."""
    checks = {}

    try:
        if postgres_client.conn_is_mock():
            raise RuntimeError("connected to in-memory fallback")
        # Do not ping through the application's long-lived shared connection:
        # a plain SELECT starts a transaction in psycopg2 and readiness probes
        # would leave it idle, retaining relation locks.  A short autocommit
        # connection keeps health checks observational and self-contained.
        health_conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname=config.POSTGRES_DB,
            connect_timeout=3,
            application_name="icp-rag-readiness",
        )
        try:
            health_conn.autocommit = True
            with health_conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        finally:
            health_conn.close()
        checks["postgresql"] = {"ok": bool(row and row[0] == 1)}
    except Exception as exc:
        checks["postgresql"] = {"ok": False, "error": str(exc)}
        try:
            postgres_client.conn.rollback()
        except Exception:
            pass

    try:
        search_store.ensure_ready()
        checks["opensearch"] = {"ok": True}
    except Exception as exc:
        checks["opensearch"] = {"ok": False, "error": str(exc)}

    try:
        from backend.storage.hugegraph_client import hugegraph_client

        checks["hugegraph"] = hugegraph_client.capability_status(timeout=3)
    except Exception as exc:
        checks["hugegraph"] = {
            "ok": False,
            "rest_projection_read": {
                "ok": False,
                "required": True,
                "error": " ".join(str(exc).split())[:240],
            },
            "gremlin_query": {"ok": False, "required": False},
        }

    artifact_dir = config.ARTIFACT_STORAGE_DIR
    checks["artifact_store"] = {
        "ok": os.path.isdir(artifact_dir) and os.access(artifact_dir, os.R_OK | os.W_OK)
    }
    return checks


@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError):
    return JSONResponse(status_code=exc.code, content=fail(exc.code, exc.message, exc.detail))


@app.on_event("startup")
def _startup():
    """初始化 ICP 专用数据表、七类类型注册、默认诊断 Agent，并恢复中断任务。"""
    if config.STRICT_DEPENDENCIES:
        checks = _dependency_status()
        failed = [name for name, result in checks.items() if not result["ok"]]
        if failed:
            raise RuntimeError(f"required dependencies are not ready: {', '.join(failed)}")

    try:
        agent_db.init_tables()
    except Exception as exc:
        print(f"[Agent] 表初始化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise RuntimeError("Agent 数据表初始化失败") from exc

    try:
        from backend.icp.repository import icp_repository
        icp_repository.init_tables()
    except Exception as exc:
        print(f"[ICP] 领域表初始化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise RuntimeError("ICP 权威事实表初始化失败") from exc

    try:
        from backend.icp.diagram_repository import diagram_repository
        diagram_repository.init_tables()
    except Exception as exc:
        print(f"[ICP] 图审核与投影表初始化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise RuntimeError("ICP 图审核与投影表初始化失败") from exc

    try:
        from backend.icp.entity_resolution import entity_resolution_repository

        entity_resolution_repository.init_tables()
    except Exception as exc:
        print(f"[ICP] 实体归一与关系断言表初始化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise RuntimeError("ICP 实体归一与关系断言表初始化失败") from exc

    try:
        from backend.icp.document_types import normalize_document_types
        normalize_document_types()
    except Exception as exc:
        print(f"[ICP] 七类文档注册表规范化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise RuntimeError("ICP 七类文档注册表规范化失败") from exc

    try:
        from backend.agent.icp_defaults import ensure_icp_diagnosis_agent
        agent_id = ensure_icp_diagnosis_agent()
        if config.STRICT_DEPENDENCIES and not agent_id:
            raise RuntimeError("默认 ICP 故障诊断智能体未完成初始化")
    except Exception as exc:
        print(f"[ICP] 默认诊断智能体初始化警告: {exc}")
        if config.STRICT_DEPENDENCIES:
            raise

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
            # SELECT-only recovery passes also start a transaction.  Close it
            # so startup never leaves an idle transaction holding table locks.
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


@app.get("/livez", tags=["health"])
async def liveness():
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readiness():
    checks = _dependency_status()
    ready = all(result["ok"] for result in checks.values())
    payload = {"status": "ready" if ready else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if ready else 503, content=payload)


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
