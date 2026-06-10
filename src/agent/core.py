"""
Agent 核心中枢模块 (ReAct Agent Core)
========================================

核心职责：
    实现 ReAct（Reasoning + Acting）循环，作为整个系统的「大脑」，
    协调 LLM 推理、知识库检索和外部工具调用。

ReAct 循环流程：
    ┌─────────────────────────────────────────┐
    │                                         │
    │  用户提问 → LLM 决策 → 执行动作 → 观察结果  │
    │                          ↑        ↓     │
    │                          └── 循环 ─┘     │
    │                     (最多 MAX_ITER 轮)    │
    │                                         │
    │              ↓ 决策为「直接回答」           │
    │         返回最终答案 + 溯源信息             │
    └─────────────────────────────────────────┘

设计原则：
    1. 工具统一抽象：RAG 检索和外部工具都通过 BaseTool 接口暴露给 LLM。
    2. 有限循环：最多迭代 MAX_ITERATIONS 轮，防止死循环。
    3. 全链路追踪：每一步 Reasoning 和 Action 都记录到 AgentTraceLogger。
    4. 优雅降级：任何步骤失败都不会导致系统崩溃，而是记录错误并继续。

在整体架构中的位置：
    这是整个项目的最顶层——它组合了 Phase 1~3 的所有产出：
    - Phase 1 (LLMClient)    → self.llm_client
    - Phase 2 (Retriever)    → RAGSearchTool
    - Phase 3 (ToolRegistry) → self.tool_registry
    - Phase 4 (Prompts)      → build_system_prompt()
    - Phase 4 (Logger)       → self.trace_logger
"""

import json
from typing import List, Optional

from src.llm.client import LLMClient
from src.rag.retriever import Retriever, SearchResult
from src.tools.base import BaseTool
from src.tools.registry import ToolRegistry
from src.tools.calculator import CalculatorTool
from src.tools.time_tool import TimeTool
from src.agent.prompts import build_system_prompt
from src.utils.logger import AgentTraceLogger


# ============================================================
# RAG 检索工具 —— 将 Retriever 包装成 LLM 可调用的工具
# ============================================================

class RAGSearchTool(BaseTool):
    """
    知识库检索工具。

    将 Phase 2 的 Retriever 包装成 BaseTool 接口，
    使 LLM 能像调用 calculator 一样调用知识库检索。

    这个设计体现了「统一抽象」的价值：
        LLM 不需要知道 RAG 和计算器在底层有多大差异，
        它只需要知道「有一个工具叫 search_knowledge_base，
        接受一个 query 参数，返回相关文档内容」。
    """

    name = "search_knowledge_base"
    description = (
        "在内部知识库中搜索与查询相关的文档内容。"
        "当你需要回答关于概念解释、技术原理、操作指南等问题时，"
        "应该先使用此工具搜索知识库。"
        "查询文本应该使用关键词或自然语言问题。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "要在知识库中搜索的查询文本。"
                    "使用与用户问题相关的关键词或短语。"
                    "例如：'向量数据库 余弦相似度'、'分块策略'"
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever):
        """
        初始化 RAG 检索工具。

        参数：
            retriever : Phase 2 构建的 Retriever 实例（已索引好文档）
        """
        super().__init__()
        self._retriever = retriever
        self._last_results: List[SearchResult] = []  # 保存最近一次搜索结果，用于溯源

    def execute(self, query: str = "") -> str:
        """
        执行知识库检索。

        参数：
            query : 搜索查询文本

        返回值：
            格式化后的检索结果文本，包含每个结果的相似度和内容摘要。
            如果没有找到相关结果，返回明确提示。
        """
        query = query.strip()

        if len(query) == 0:
            return "[知识库] 查询文本为空，请提供有效的搜索关键词。"

        # 执行检索（top_k=3，相似度阈值=0.3）
        results = self._retriever.search(
            query=query,
            top_k=3,
            score_threshold=0.3,  # 过滤掉相似度太低的噪音结果
        )

        # 保存结果供后续溯源
        self._last_results = results

        # 格式化结果返回给 LLM
        if len(results) == 0:
            return "[知识库] 未找到与查询相关的文档内容。请告知用户知识库中没有相关信息。"

        formatted_parts = []
        for i, result in enumerate(results):
            source = result.metadata.get("file_name", "未知来源")
            score = result.score
            content = result.chunk_content[:300]  # 截断过长内容
            formatted_parts.append(
                f"--- 结果 {i+1}（相似度: {score:.2%}，来源: {source}）---\n"
                f"{content}"
            )

        return "\n\n".join(formatted_parts)

    def get_last_sources(self) -> List[str]:
        """
        获取最近一次检索的引用来源列表。
        用于最终回答时标注出处。
        """
        sources = []
        seen = set()
        for r in self._last_results:
            source = r.metadata.get("file_name", "未知来源")
            if source not in seen:
                sources.append(source)
                seen.add(source)
        return sources


# ============================================================
# ReAct Agent
# ============================================================

class ReActAgent:
    """
    ReAct Agent —— 整个项目的核心中枢。

    使用方法：
        agent = ReActAgent()
        agent.index_documents(["data/文档1.md", "data/文档2.pdf"])

        answer, trace = agent.run("什么是向量数据库？")
        print(answer)
        trace.print_summary()
    """

    # 最大推理迭代次数（防止死循环）
    MAX_ITERATIONS = 5

    def __init__(self):
        """
        初始化 Agent。

        这一步会组装所有子组件：
            - LLM 客户端（Phase 1）
            - 向量检索引擎（Phase 2）
            - 工具注册中心（Phase 3）
            - 日志追踪器（Phase 4）
        """
        print("[Agent] 正在初始化 ReAct Agent...")

        # 核心子组件
        self.llm_client = LLMClient()           # Phase 1
        self.retriever = Retriever()             # Phase 2
        self.trace_logger = AgentTraceLogger()   # Phase 4

        # 将 RAG 检索包装为工具
        self.rag_tool = RAGSearchTool(self.retriever)

        # 工具注册中心（Phase 3）
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(self.rag_tool)         # 知识库检索
        self.tool_registry.register(CalculatorTool())       # 计算器
        self.tool_registry.register(TimeTool())             # 时间查询

        print(f"   [OK] Agent 初始化完成。已加载 {len(self.tool_registry.list_tools())} 个工具。")

    # ============================================================
    # 文档索引
    # ============================================================

    def index_documents(self, file_paths: List[str]) -> int:
        """
        索引一批文档（解析 + 分块 + 向量化 + 存入 ChromaDB）。

        这是 Agent 的「知识摄入」入口——在回答用户问题之前，
        先把知识库文档加载到向量库中。

        参数：
            file_paths : 文档路径列表

        返回值：
            成功索引的总 Chunk 数
        """
        from src.rag.parser import DocumentParser
        parser = DocumentParser()

        total_chunks = 0
        for file_path in file_paths:
            print(f"\n[Doc] 正在索引文档：{file_path}")
            try:
                document = parser.parse(file_path)
                chunk_count = self.retriever.index(document)
                total_chunks += chunk_count
            except Exception as error:
                print(f"   [FAIL] 索引失败：{error}")

        print(f"\n[Stats] 索引完成：共 {total_chunks} 个 Chunk 入库。")
        return total_chunks

    # ============================================================
    # 核心方法：运行 Agent
    # ============================================================

    def run(self, user_question: str) -> dict:
        """
        执行 ReAct 循环，回答用户问题。

        这是整个项目最核心的方法——它实现了完整的 Reasoning + Acting 循环。

        参数：
            user_question : 用户的问题文本

        返回值：
            一个字典，包含：
                - "answer"  : str —— 最终回答文本
                - "sources" : list —— 引用来源列表
                - "trace"   : AgentTraceLogger —— 完整追踪日志
                - "iterations" : int —— 使用的推理迭代次数
        """
        # 每次调用创建新的追踪日志
        self.trace_logger = AgentTraceLogger()

        print(f"\n{'='*60}")
        print(f"[Start] Agent 收到问题：{user_question}")
        print(f"{'='*60}")

        # --- 构建初始消息 ---
        system_prompt = build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        # 获取所有工具的 JSON Schema
        tools = self.tool_registry.get_schemas()

        final_answer = ""
        sources = []

        # --- ReAct 主循环 ---
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n─ 迭代 {iteration}/{self.MAX_ITERATIONS} ─")

            # 第一步：调用 LLM
            try:
                response = self.llm_client._invoke_with_retry(
                    messages=messages,
                    temperature=0.0,  # 低温度 = 更稳定的决策
                    tools=tools,
                )
            except Exception as error:
                self.trace_logger.log_error(f"LLM 调用失败：{error}")
                final_answer = f"抱歉，系统在处理您的问题时遇到了错误。请稍后重试。"
                break

            choice = response.choices[0]
            message = choice.message

            # 第二步：判断 LLM 的决策

            # 情况 A：LLM 决定调用工具
            if message.tool_calls and len(message.tool_calls) > 0:
                # 记录推理过程（从 finish_reason 推断）
                self.trace_logger.log_reasoning(
                    f"LLM 决定调用 {len(message.tool_calls)} 个工具来获取信息。"
                )

                # 构建 assistant 消息（包含完整的 tool_calls）
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                messages.append(assistant_message)

                # 逐个执行工具调用
                for tc in message.tool_calls:
                    tool_name = tc.function.name

                    # 解析参数（JSON 字符串 → Python 字典）
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # 执行工具
                    if tool_name == "search_knowledge_base":
                        # RAG 检索：通过 rag_tool 执行并记录
                        result = self.rag_tool.execute(**arguments)
                        self.trace_logger.log_rag_search(
                            query=arguments.get("query", ""),
                            results=self.rag_tool._last_results,
                        )
                        # 收集溯源信息
                        for s in self.rag_tool.get_last_sources():
                            if s not in sources:
                                sources.append(s)
                    else:
                        # 外部工具：通过 Registry 执行并记录
                        result = self.tool_registry.execute(tool_name, arguments)
                        self.trace_logger.log_tool_call(
                            tool_name=tool_name,
                            arguments=arguments,
                            result=result,
                        )

                    # 将工具结果加入对话历史（tool role）
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                # 继续下一轮迭代（LLM 综合工具结果后决定下一步）
                continue

            # 情况 B：LLM 决定直接回答（没有 tool_calls）
            else:
                final_answer = message.content or ""

                # 检查是否触发了「拒绝回答」
                if self._is_refusal(final_answer):
                    self.trace_logger.log_refuse(
                        "Agent 判断无法回答此问题（知识库无相关内容或能力不足）。"
                    )
                else:
                    self.trace_logger.log_reasoning(
                        "LLM 认为已有足够信息，决定直接回答。"
                    )

                break

        # --- 循环结束 ---

        # 如果循环耗尽仍未得到最终答案
        if not final_answer:
            final_answer = (
                "抱歉，经过多轮检索和工具调用，仍然无法完全回答您的问题。"
                "这可能意味着知识库中缺少相关信息。"
            )
            self.trace_logger.log_refuse("推理迭代耗尽，未能得出确定答案。")

        # 记录最终回答
        self.trace_logger.log_final_answer(final_answer, sources)

        return {
            "answer": final_answer,
            "sources": sources,
            "trace": self.trace_logger,
            "iterations": min(iteration, self.MAX_ITERATIONS),
        }

    # ============================================================
    # 内部辅助方法
    # ============================================================

    def _is_refusal(self, text: str) -> bool:
        """
        检测回答是否为「拒绝回答」模式。

        通过关键词匹配判断 LLM 是否在表达「不知道」。
        这是一个简单但有效的启发式方法，
        结合 Prompt 中的「拒绝回答」指令形成双层防护。

        参数：
            text : LLM 输出的回答文本

        返回值：
            True 表示这是一个拒绝回答
        """
        if not text:
            return True

        # 拒绝回答的关键词特征
        refusal_keywords = [
            "无法回答", "无法确定", "不太确定", "我不能",
            "知识库中没有", "没有找到相关", "未找到相关",
            "抱歉", "对不起，我无法",
            "超出", "能力范围",
        ]

        text_lower = text.lower()
        for keyword in refusal_keywords:
            if keyword in text_lower:
                return True

        return False
