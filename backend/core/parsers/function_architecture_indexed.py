"""Function Locator 状态适配。

系统架构图的结构化结果已直接服务于 Function Locator，不要求人工再生成 HugeGraph，
因此解析完成后保持 indexed，避免出现无法执行的 kg_pending 状态。
"""
from backend.core.parsers.function_architecture import FunctionArchitectureParser
from backend.core.parsers.registry import ParserRegistry


@ParserRegistry.register("function_locator")
class IndexedFunctionArchitectureParser(FunctionArchitectureParser):
    def parse(self, file_path, ctx):
        result = super().parse(file_path, ctx)
        result.stats["kg_pending"] = False
        return result
