"""图谱层：re-export app.graphrag 现有实现（图谱抽取与构建）。

不复制逻辑，仅引用，确保功能与现有版本完全一致。
"""
from backend.graphrag.graph_builder import GraphBuilder  # noqa: F401
from backend.graphrag.graph_extractor import GraphExtractor  # noqa: F401
