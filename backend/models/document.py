"""映射现有 documents / document_chunks 表（结构零改动）。"""
from datetime import datetime

from sqlalchemy import String, Text, Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255))
    kb_id: Mapped[str] = mapped_column(String(36))
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    # 物理列在兼容期继续使用 minio_path；业务代码通过 storage_key 访问中性名称。
    minio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    delete_step: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @property
    def storage_key(self) -> str | None:
        return self.minio_path

    @storage_key.setter
    def storage_key(self, value: str | None) -> None:
        self.minio_path = value


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(36))
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str] = mapped_column(String(32), default="text")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    selected_for_kg: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
