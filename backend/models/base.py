"""SQLAlchemy 声明式基类（映射现有表，不重建、不迁移）。"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
