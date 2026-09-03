"""Function Locator 状态适配。

确定性系统/模块/功能/接口由 PostgreSQL 直接服务；架构图片仅保存为待复核候选，
当前阶段不发布 HugeGraph。因此解析完成后保持 indexed，避免出现无法执行的
kg_pending 状态。
"""
from backend.core.parsers.function_architecture import FunctionArchitectureParser
from backend.core.parsers.registry import ParserRegistry


@ParserRegistry.register("function_locator")
class IndexedFunctionArchitectureParser(FunctionArchitectureParser):
    def parse(self, file_path, ctx):
        result = super().parse(file_path, ctx)
        result.stats["kg_pending"] = False
        return result
