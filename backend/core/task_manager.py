import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from backend.storage.postgres_client import postgres_client

@dataclass
class TaskState:
    task_id: str
    doc_id: str
    kb_id: str
    file_name: str
    task_type: str
    status: str = "pending"
    progress: int = 0
    logs: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    cancel_requested: bool = False

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self.task_queues: Dict[str, asyncio.Queue] = {}
        self._lock = asyncio.Lock()

    async def create_task(self, doc_id: str, kb_id: str, file_name: str, task_type: str) -> str:
        task_id = str(uuid.uuid4())
        async with self._lock:
            self.tasks[task_id] = TaskState(
                task_id=task_id,
                doc_id=doc_id,
                kb_id=kb_id,
                file_name=file_name,
                task_type=task_type,
                start_time=datetime.now()
            )
            self.task_queues[task_id] = asyncio.Queue()
        return task_id

    async def update_progress(self, task_id: str, progress: int, log_message: str = "", 
                             log_type: str = "general", chunk_id: str = None, chunk_index: int = 0, 
                             chunk_title: str = None, entity_count: int = 0, relation_count: int = 0,
                             entities: Optional[Dict] = None, relations: Optional[Dict] = None,
                             block_key: str = None, block_title: str = None,
                             block_count: int = 0, image_count: int = 0, table_count: int = 0,
                             community_count: int = 0):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = max(0, min(100, progress))
                if log_message:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.tasks[task_id].logs.append(f"[{timestamp}] {log_message}")
                    await self.task_queues[task_id].put({
                        "type": "progress",
                        "progress": progress,
                        "log": log_message,
                        "log_type": log_type
                    })
                
                try:
                    task = self.tasks.get(task_id)
                    if task:
                        log_id = str(uuid.uuid4())
                        if task.task_type == "parse":
                            postgres_client.insert_parse_log(
                                log_id=log_id,
                                task_id=task_id,
                                doc_id=task.doc_id,
                                kb_id=task.kb_id,
                                block_key=block_key,
                                block_title=block_title,
                                log_level="INFO",
                                log_type=log_type,
                                message=log_message,
                                block_count=block_count,
                                image_count=image_count,
                                table_count=table_count,
                                progress=progress,
                                status=task.status
                            )
                        else:
                            postgres_client.insert_kg_log(
                                log_id=log_id,
                                task_id=task_id,
                                doc_id=task.doc_id,
                                kb_id=task.kb_id,
                                chunk_id=chunk_id,
                                chunk_index=chunk_index,
                                chunk_title=chunk_title,
                                log_level="INFO",
                                log_type=log_type,
                                message=log_message,
                                entity_count=entity_count,
                                relation_count=relation_count,
                                community_count=community_count,
                                entities=entities,
                                relations=relations,
                                progress=progress,
                                status=task.status
                            )
                except Exception as e:
                    try:
                        postgres_client.conn.rollback()
                    except Exception:
                        pass
                    print(f"Failed to persist log: {e}", flush=True)

    async def cancel_task(self, task_id: str) -> bool:
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in ["completed", "failed", "cancelled"]:
                    return False
                task.cancel_requested = True
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.logs.append(f"[{timestamp}] [CANCEL] 用户请求取消任务")
                await self.task_queues[task_id].put({
                    "type": "cancel_requested"
                })
                return True
            return False

    async def is_cancelled(self, task_id: str) -> bool:
        async with self._lock:
            task = self.tasks.get(task_id)
            if task:
                return task.cancel_requested
            return False

    async def check_cancelled(self, task_id: str):
        async with self._lock:
            task = self.tasks.get(task_id)
            if task and task.cancel_requested:
                task.status = "cancelled"
                task.end_time = datetime.now()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                task.logs.append(f"[{timestamp}] [CANCEL] 任务已取消")
                await self.task_queues[task_id].put({
                    "type": "status",
                    "status": "cancelled"
                })
                raise asyncio.CancelledError(f"Task {task_id} cancelled by user")

    async def update_status(self, task_id: str, status: str):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = status
                if status in ["completed", "failed", "cancelled"]:
                    self.tasks[task_id].end_time = datetime.now()
                    await self.task_queues[task_id].put({
                        "type": "status",
                        "status": status
                    })

    async def set_result(self, task_id: str, result: Dict):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].result = result

    async def set_error(self, task_id: str, error: str):
        async with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].error = error
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.tasks[task_id].logs.append(f"[{timestamp}] ERROR: {error}")
                await self.task_queues[task_id].put({
                    "type": "error",
                    "error": error
                })
                
                try:
                    task = self.tasks.get(task_id)
                    if task:
                        log_id = str(uuid.uuid4())
                        if task.task_type == "parse":
                            postgres_client.insert_parse_log(
                                log_id=log_id,
                                task_id=task_id,
                                doc_id=task.doc_id,
                                kb_id=task.kb_id,
                                log_level="ERROR",
                                log_type="error",
                                message=f"ERROR: {error}",
                                progress=task.progress,
                                status="failed"
                            )
                        else:
                            postgres_client.insert_kg_log(
                                log_id=log_id,
                                task_id=task_id,
                                doc_id=task.doc_id,
                                kb_id=task.kb_id,
                                log_level="ERROR",
                                log_type="error",
                                message=f"ERROR: {error}",
                                progress=task.progress,
                                status="failed"
                            )
                except Exception as e:
                    try:
                        postgres_client.conn.rollback()
                    except Exception:
                        pass
                    print(f"Failed to persist error log: {e}", flush=True)

    async def get_task(self, task_id: str) -> Optional[TaskState]:
        async with self._lock:
            return self.tasks.get(task_id)

    async def get_tasks_by_doc(self, doc_id: str) -> List[TaskState]:
        async with self._lock:
            return [task for task in self.tasks.values() if task.doc_id == doc_id]

    async def remove_task(self, task_id: str):
        async with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
            if task_id in self.task_queues:
                del self.task_queues[task_id]

    async def stream_events(self, task_id: str):
        if task_id not in self.task_queues:
            return
        
        queue = self.task_queues[task_id]
        
        async with self._lock:
            task = self.tasks.get(task_id)
            if task:
                for log in task.logs:
                    yield {"type": "log", "log": log}
                yield {"type": "progress", "progress": task.progress, "status": task.status}
        
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield event
                
                async with self._lock:
                    task = self.tasks.get(task_id)
                    if task and task.status in ["completed", "failed", "cancelled"]:
                        queue.task_done()
                        return
            except asyncio.TimeoutError:
                async with self._lock:
                    task = self.tasks.get(task_id)
                    if not task or task.status in ["completed", "failed", "cancelled"]:
                        return

task_manager = TaskManager()