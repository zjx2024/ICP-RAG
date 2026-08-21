"""后端业务路由聚合。

Dify 外部知识库接口保持原路径和行为；新增 ICP 七类文档与故障诊断领域接口。
Agent 路由由 backend.main 单独挂载到 /api 前缀。
"""
from fastapi import APIRouter

from backend.api.routers.documents import router as documents_router
from backend.api.routers.models import router as models_router
from backend.api.routers.retrieval import router as retrieval_router
from backend.api.routers.dify import router as dify_router
from backend.api.routers.types import router as types_router
from backend.api.routers.icp_domain import router as icp_domain_router

router = APIRouter()
router.include_router(retrieval_router, tags=["知识库检索"])
router.include_router(documents_router, tags=["文档管理"])
router.include_router(models_router, tags=["模型管理"])
router.include_router(dify_router, tags=["Dify 对接"])
router.include_router(types_router, tags=["ICP 文档类型"])
router.include_router(icp_domain_router, tags=["ICP 故障诊断领域"])
