"""配置层：复用 app.config.config 单例（单一实现来源）。

重构原则：不复制逻辑，仅 re-export，确保"功能不受丝毫影响"。
"""
from backend.config.config import config  # noqa: F401
