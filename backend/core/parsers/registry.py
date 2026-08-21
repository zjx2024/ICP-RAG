"""ParserRegistry（策略模式）：按 doc_type 注册/获取解析器。"""
from typing import Dict

from backend.core.parsers.base import Parser


class ParserRegistry:
    _parsers: Dict[str, Parser] = {}

    @classmethod
    def register(cls, doc_type: str):
        """装饰器：`@ParserRegistry.register('hmc')` 将解析器类注册到 doc_type。"""
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


# 触发各解析器模块完成注册（模块底部导入避免循环依赖）
from backend.core.parsers import excel_row, fault_tree, text  # noqa: E402,F401
