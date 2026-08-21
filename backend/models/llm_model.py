"""映射现有 llm_models 表（结构零改动，模型身份唯一来源）。"""
from datetime import datetime

from sqlalchemy import String, Text, Float, Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class LLMModel(Base):
    __tablename__ = "llm_models"

    model_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    model_type: Mapped[str] = mapped_column(String(30), default="llm")
    provider: Mapped[str] = mapped_column(String(50))
    api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=262144)
    top_p: Mapped[float] = mapped_column(Float, default=0.1)
    timeout: Mapped[int] = mapped_column(Integer, default=3000)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
