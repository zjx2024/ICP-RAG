from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import socket
from backend.core.retrieval import knowledge_retriever
from backend.storage.postgres_client import postgres_client

router = APIRouter(prefix="/dify", tags=["dify"])


@router.get("/server-info")
async def get_server_info():
    """返回服务器局域网 IP，供前端拼接 Dify API 地址

    检测顺序（可靠性从高到低）：
      1. SERVER_IP 环境变量（由 start.sh 自动检测并注入，或用户手动指定）
      2. socket 探测（容器内通常是容器 IP，不可靠，仅作兜底）
    返回字段:
      - lan_ip: 检测到的 IP
      - source: IP 来源（"env" / "auto"），前端可据此判断是否需要提示用户
    """
    lan_ip = os.environ.get("SERVER_IP", "").strip()
    if lan_ip:
        return {"lan_ip": lan_ip, "source": "env"}

    # 兜底：socket 探测（容器内拿到的是容器 IP，可能无法被局域网访问）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"
    return {"lan_ip": lan_ip, "source": "auto"}


class DifyRetrievalSetting(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0


class DifyRetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: DifyRetrievalSetting
    metadata_condition: Optional[Dict[str, Any]] = None


class DifyRecord(BaseModel):
    content: str
    score: float
    title: str
    metadata: Dict[str, Any] = {}


@router.post("/retrieval")
async def dify_retrieval(
    request: DifyRetrievalRequest,
    authorization: Optional[str] = Header(None),
):
    """Dify 外部知识库 API 兼容端点

    Dify 注册时的 API 接口地址填: http://<host>:9999/api/v1/dify
    Dify 会自动追加 /retrieval 到此地址。
    """
    kb_id = request.knowledge_id
    query = request.query
    top_k = request.retrieval_setting.top_k
    score_threshold = request.retrieval_setting.score_threshold

    # 校验知识库存在
    kb = postgres_client.get_knowledge_base(kb_id)
    if not kb:
        return {"error_code": 2001, "error_msg": f"Knowledge base not found: {kb_id}"}

    # 使用 RAG 融合检索（文档分块 + 知识图谱实体 + 社区报告）
    try:
        result = knowledge_retriever.search_rag_fusion(kb_id, query, top_k=top_k * 2)
    except Exception as e:
        return {"error_code": 500, "error_msg": f"Search failed: {str(e)}"}

    records: List[Dict[str, Any]] = []

    # 1. 文档分块
    for chunk in result.get("chunks", []):
        score = float(chunk.get("score", 0.0))
        if score_threshold > 0 and score < score_threshold:
            continue
        content = chunk.get("content", "")
        if not content:
            continue
        records.append({
            "content": content,
            "score": min(score / 20, 1.0) if score > 1 else score,
            "title": chunk.get("title", "") or chunk.get("doc_name", ""),
            "metadata": {
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "section": chunk.get("section", ""),
                "content_type": chunk.get("content_type", "text"),
                "data_type": "chunk",
            },
        })

    # 2. 知识图谱实体
    for entity in result.get("entities", []):
        score = float(entity.get("score", 0.0))
        if score_threshold > 0 and score < score_threshold:
            continue
        name = entity.get("entity_name", "")
        desc = entity.get("description", "")
        if not name and not desc:
            continue
        content = f"[实体] {name}（{entity.get('entity_type', '')}）\n{desc}" if desc else f"[实体] {name}（{entity.get('entity_type', '')}）"
        records.append({
            "content": content,
            "score": min(score, 1.0) if score <= 1 else min(score / 20, 1.0),
            "title": name,
            "metadata": {
                "entity_id": entity.get("entity_id", ""),
                "entity_type": entity.get("entity_type", ""),
                "doc_id": entity.get("doc_id", ""),
                "weight": entity.get("weight", 0),
                "data_type": "entity",
            },
        })

    # 3. 社区报告
    for report in result.get("community_reports", []):
        score = float(report.get("score", 0.0))
        if score_threshold > 0 and score < score_threshold:
            continue
        content = report.get("content", "") or report.get("report", "")
        if not content:
            continue
        records.append({
            "content": content,
            "score": min(score / 20, 1.0) if score > 1 else score,
            "title": report.get("title", "") or f"社区报告-{report.get('community_id', '')}",
            "metadata": {
                "report_id": report.get("report_id", ""),
                "community_id": report.get("community_id", ""),
                "doc_id": report.get("doc_id", ""),
                "data_type": "community_report",
            },
        })

    # 按 score 降序排序，取 top_k
    records.sort(key=lambda x: x["score"], reverse=True)
    records = records[:top_k]

    return {"records": records}
