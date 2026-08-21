"""解析器统一接口与上下文/结果定义（ICP 类型驱动解析架构）。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ParseContext:
    """解析上下文：由统一编排注入，解析器只依赖这些信息。"""
    doc_id: str
    doc_type: str
    file_name: str
    task_id: str = ""
    model_config: Optional[Dict] = None
    temp_path: str = ""
    kb_id: str = ""          # 兼容：图类 document_fault_trees.kb_id 列语义存 doc_type
    progress_cb: Optional[Callable[[str], None]] = None  # SSE 进度回调（可选）

    def progress(self, message: str):
        if self.progress_cb:
            self.progress_cb(message)


@dataclass
class ParseResult:
    """解析结果统计（落 icp_documents.last_result）。"""
    doc_id: str
    doc_type: str
    stats: Dict = field(default_factory=dict)   # rows_ok/rows_failed/failed_rows/nodes/edges/chunks/vectors
    vectors: int = 0
    message: str = ""


class Parser(ABC):
    """解析器基类：所有 ICP 解析器继承，由 ParserRegistry 按 doc_type 路由。"""
    doc_type: str = ""

    @abstractmethod
    def validate(self, file_path: str) -> List[str]:
        """格式/表头校验，返回错误列表（空列表 = 校验通过）。"""

    @abstractmethod
    def parse(self, file_path: str, ctx: ParseContext) -> ParseResult:
        """执行解析并入库（PG + ES），返回统计。"""
