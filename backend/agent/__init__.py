"""Agent 模块：基于 LangGraph 的智能体平台扩展。

本模块为现有知识图谱/RAG 项目的扩展，零侵入设计：
- 不修改任何现有表结构（包括 llm_models）
- 不修改任何现有代码文件的已有逻辑
- 所有 Agent 数据存入新建表（agent_models/agent_configs/agent_sessions/agent_messages）
- 所有 Agent 代码集中在 app/agent/ 目录
- 现有 core/、storage/、api/routers/ 保持不变，仅只读引用
"""
