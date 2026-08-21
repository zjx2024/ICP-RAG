"""DeletionLogger: 文档删除过程的结构化日志记录器。

把删除操作的每一步（删边、删顶点、媒体、PG 分块、日志、ES、MinIO 文件、PG 记录）
都以结构化的方式写入 PostgreSQL `deletion_logs` 表，供前端在"日志管理"中查看。
"""
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class DeletionLogger:
    """围绕一次文档删除任务（task_id）的日志收集器。

    使用方式：
        logger = DeletionLogger(postgres_client, kb_id, doc_id)
        logger.log_step_start("edges")
        logger.log_item("edges", "edge", edge_id, details={...})
        logger.log_item("edges", "edge", edge_id, status="deleted", ...)
        logger.log_step_complete("edges", count=10)
        logger.finalize(status="success")

    task_id 用于把同一个删除任务的所有日志关联起来。
    """

    STEPS = [
        "edges",
        "vertices",
        "media",
        "chunks",
        "logs",
        "es",
        "file",
        "pg_record",
        "done",
    ]

    def __init__(self, postgres_client, kb_id: str, doc_id: str, task_id: Optional[str] = None,
                 kb_name: Optional[str] = None, doc_name: Optional[str] = None):
        self.pg = postgres_client
        self.kb_id = kb_id
        self.doc_id = doc_id
        self.task_id = task_id or str(uuid.uuid4())
        # 来源名字快照：删除流程后期文档记录会被删掉，这里先记住名字写入日志留痕
        self.kb_name = kb_name
        self.doc_name = doc_name
        self._buffer: List[Dict[str, Any]] = []
        self._step_start_time: Dict[str, str] = {}
        logger.info(f"DeletionLogger initialized: task_id={self.task_id}, kb_id={kb_id}, doc_id={doc_id}")

    def log_step_start(self, step: str, message: Optional[str] = None):
        """记录某一步骤开始。"""
        msg = message or f"步骤 '{step}' 开始"
        self._write_log(
            step=step,
            item_type="step",
            item_id="start",
            status="info",
            message=msg,
        )
        self._step_start_time[step] = datetime.now().isoformat()

    def log_step_complete(self, step: str, count: int = 0, errors: int = 0, message: Optional[str] = None):
        """记录某一步骤完成，附带统计信息。"""
        msg = message or f"步骤 '{step}' 完成: 处理 {count} 条，{errors} 个错误"
        status = "error" if errors > 0 else "success"
        self._write_log(
            step=step,
            item_type="step",
            item_id="complete",
            item_detail={"count": count, "errors": errors},
            status=status,
            message=msg,
        )

    def log_item(self, step: str, item_type: str, item_id: str,
                 status: str = "info", message: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """记录单个条目的处理（删除/更新/跳过等）。"""
        self._write_log(
            step=step,
            item_type=item_type,
            item_id=item_id,
            item_detail=details,
            status=status,
            message=message,
        )

    def log_summary(self, summary: Dict[str, Any]):
        """记录删除任务的最终汇总。"""
        self._write_log(
            step="done",
            item_type="summary",
            item_id="summary",
            item_detail=summary,
            status="success" if summary.get("errors", 0) == 0 else "warning",
            message=f"文档删除完成: {summary}",
        )

    def _write_log(self, step: str, item_type: str, item_id: Optional[str] = None,
                   item_detail: Optional[Dict[str, Any]] = None,
                   status: str = "info", message: Optional[str] = None):
        """写入一条日志记录。优先直接写库，失败时回落到内存 buffer。"""
        try:
            self.pg.insert_deletion_log(
                task_id=self.task_id,
                doc_id=self.doc_id,
                kb_id=self.kb_id,
                step=step,
                item_type=item_type,
                item_id=item_id,
                item_detail=item_detail,
                status=status,
                message=message,
                kb_name=self.kb_name,
                doc_name=self.doc_name,
            )
            logger.debug(f"Deletion log written: task_id={self.task_id}, step={step}, item_type={item_type}, item_id={item_id}, status={status}")
        except Exception as e:
            logger.warning(f"Failed to write deletion log to DB, buffering: task_id={self.task_id}, step={step}, error={str(e)}")
            # 数据库不可用时先缓存在内存
            self._buffer.append({
                "task_id": self.task_id,
                "doc_id": self.doc_id,
                "kb_id": self.kb_id,
                "step": step,
                "item_type": item_type,
                "item_id": item_id,
                "item_detail": item_detail,
                "status": status,
                "message": message,
                "buffered_at": datetime.now().isoformat(),
            })

    @property
    def buffered_logs(self) -> List[Dict[str, Any]]:
        """返回因数据库异常而缓存的日志（供后续重试或诊断）。"""
        return list(self._buffer)
