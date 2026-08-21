"""ParserRegistry（策略模式）：按七类 ICP 文档注册/获取专用解析器。"""
from typing import Dict

from backend.core.parsers.base import Parser


class ParserRegistry:
    _parsers: Dict[str, Parser] = {}

    @classmethod
    def register(cls, doc_type: str):
        def deco(parser_cls):
            instance = parser_cls()
            instance.doc_type = doc_type
            cls._parsers[doc_type] = instance
            return parser_cls
        return deco

    @classmethod
    def get(cls, doc_type: str) -> Parser:
        parser = cls._parsers.get(doc_type)
        if parser is None:
            raise KeyError(f"未注册文档类型解析器: {doc_type}")
        return parser

    @classmethod
    def has(cls, doc_type: str) -> bool:
        return doc_type in cls._parsers

    @classmethod
    def keys(cls):
        return list(cls._parsers.keys())


# 导入即完成注册：HMC/FMECA/MFL、FTA/RBD、故障历史、系统架构与功能。
from backend.core.parsers import (  # noqa: E402,F401
    excel_row,
    diagram,
    fault_history,
    function_architecture,
)
