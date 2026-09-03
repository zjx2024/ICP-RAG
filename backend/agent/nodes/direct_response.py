"""无需工程资料检索的统一产品响应。"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from backend.agent.nodes.scope_router import (
    SCOPE_AGENT_CAPABILITY,
    SCOPE_CLARIFICATION,
    SCOPE_OUT_OF_SCOPE,
)


def _capability_response(agent_name: str) -> str:
    return f"""我是 **{agent_name}**，面向 ICP 航空电子系统的工程知识查询与辅助故障诊断。

我基于已经上传并解析的七类资料工作：系统架构与功能说明、HMC、FMECA、FTA、RBD、MFL 和故障历史。我可以：

- 查询系统、模块、功能、接口和业务编号的基础信息；
- 从故障现象分析可能原因、影响、检查与维修资料；
- 补充有明确来源的历史处理时间、操作步骤、最终方案和验证结果；
- 在需要跨文档关系时自动查询受控的多跳关系；
- 对通过数据门禁的 FTA/RBD 结构执行确定性计算。

你可以直接描述现象、发生阶段、告警文字和已执行操作，也可以提供模块名称、`FUNC`、`LCN/HAMC/HSC` 或 HMC 编号。"""


def _out_of_scope_response(agent_name: str) -> str:
    return f"""这个问题不属于当前 ICP 航空电子系统知识库的范围，因此 **{agent_name}** 本轮没有检索工程资料。

我主要回答系统架构与功能、故障代码、故障原因与影响、维修定位、可靠性结构和历史排故问题。你可以改为描述相关系统、模块、`FUNC/HMC` 编号或观察到的故障现象。"""


def direct_response_node(state: Mapping[str, Any], config=None) -> Dict[str, Any]:
    route = str(state.get("scope_route") or SCOPE_OUT_OF_SCOPE)
    agent_name = str(state.get("agent_name") or "ICP故障诊断智能体").strip()
    if route == SCOPE_AGENT_CAPABILITY:
        answer = _capability_response(agent_name)
    elif route == SCOPE_CLARIFICATION:
        question = str(state.get("clarification_question") or "").strip()
        answer = question or "请说明你指的是哪个系统、模块、代码或故障现象。"
    else:
        answer = _out_of_scope_response(agent_name)

    return {
        "final_answer": answer,
        "citations": [],
        "evidence_catalog": [],
        "used_evidence_ids": [],
        "authorized_tools": [],
        "allowed_tools": [],
        "tools_plan": [],
        "tool_calls": [],
        "tool_results": [],
        "evidence_registry": [],
    }


__all__ = ["direct_response_node"]
