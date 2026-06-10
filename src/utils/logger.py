"""
Agent 追踪日志模块 (Trace Logger)
===================================

核心职责：
    记录 Agent 每一步的 Reasoning（推理摘要）、Action（工具调用入参/出参）
    和 Final Answer（最终结果），满足验收标准 4「全链路追踪」。

设计原则：
    1. 结构化记录：每一步都是独立的事件，带有时间戳和事件类型。
    2. 非侵入式：Agent 代码通过简单的方法调用记录日志，不改变业务逻辑。
    3. 可读性：日志输出人对友好，方便调试和 Prompt 迭代。

事件类型说明：
    - "reasoning"   : Agent 的推理过程（思考步骤）
    - "rag_search"  : RAG 检索（查询文本 + 检索结果）
    - "tool_call"   : 工具调用（工具名 + 入参 + 出参）
    - "final_answer": 最终回答
    - "refuse"      : 拒绝回答（知识库无相关内容）
    - "error"       : 异常错误
"""

import json
import sys
from datetime import datetime
from typing import Any, List, Optional


# 安全打印函数 —— 处理 Windows GBK 终端无法编码 Unicode 字符的问题
def _safe_print(message: str):
    """打印消息，自动处理编码错误（替换无法编码的字符为 ?）"""
    try:
        print(message)
    except UnicodeEncodeError:
        # 对 GBK 无法编码的字符进行转义处理
        safe_message = message.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(safe_message)


class TraceEvent:
    """
    单条追踪事件。

    属性：
        event_type : 事件类型（reasoning / rag_search / tool_call / final_answer / refuse / error）
        timestamp  : 事件发生时间
        data       : 事件的详细数据（字典）
    """

    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.timestamp = datetime.now().isoformat()
        self.data = data

    def to_dict(self) -> dict:
        """将事件转为字典（方便序列化为 JSON）"""
        return {
            "type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def __repr__(self):
        """事件的可读表示"""
        return f"[{self.timestamp}] {self.event_type}: {json.dumps(self.data, ensure_ascii=False, default=str)}"


class AgentTraceLogger:
    """
    Agent 追踪日志器。

    维护一个事件列表，提供各类型事件的便捷记录方法。

    使用方法：
        logger = AgentTraceLogger()
        logger.log_reasoning("需要先检索知识库获取公式，再计算数值。")
        logger.log_rag_search("余弦相似度公式", [result1, result2])
        logger.log_tool_call("calculator", {"expression": "2+3"}, "5")
        logger.log_final_answer("2+3的结果是5。", sources=["calculator"])
        logger.print_summary()  # 打印完整追踪报告
    """

    def __init__(self):
        """初始化空的事件列表"""
        self.events: List[TraceEvent] = []
        self._step_counter = 0  # 步骤计数器

    def _next_step(self) -> int:
        """递增步骤计数器并返回当前步骤号"""
        self._step_counter += 1
        return self._step_counter

    def log_reasoning(self, thought: str):
        """
        记录 Agent 的推理步骤（Thought）。

        参数：
            thought : 推理摘要 —— Agent 在这一步想了什么、为什么这么做
        """
        step = self._next_step()
        event = TraceEvent("reasoning", {
            "step": step,
            "thought": thought,
        })
        self.events.append(event)
        _safe_print(f"\n[Think] 步骤 {step} - 推理: {thought}")

    def log_rag_search(self, query: str, results: list):
        """
        记录一次 RAG 检索。

        参数：
            query   : 向向量库发送的查询文本
            results : 检索返回的 SearchResult 列表
        """
        step = self._next_step()
        results_summary = []
        for r in results:
            results_summary.append({
                "source": r.metadata.get("file_name", "?"),
                "chunk_index": r.metadata.get("chunk_index", "?"),
                "score": round(r.score, 4),
                "preview": r.chunk_content[:80],
            })

        event = TraceEvent("rag_search", {
            "step": step,
            "query": query,
            "result_count": len(results),
            "results": results_summary,
        })
        self.events.append(event)
        _safe_print(f"[Search] 步骤 {step} - RAG检索: '{query}' -> {len(results)}个结果")

    def log_tool_call(self, tool_name: str, arguments: dict, result: str):
        """
        记录一次工具调用。

        参数：
            tool_name : 被调用的工具名称
            arguments : 工具的入参
            result    : 工具的返回值
        """
        step = self._next_step()
        event = TraceEvent("tool_call", {
            "step": step,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        })
        self.events.append(event)
        _safe_print(f"[Tool] 步骤 {step} - 工具调用: {tool_name}({arguments}) -> {result[:100]}")

    def log_final_answer(self, answer: str, sources: list = None):
        """
        记录最终回答。

        参数：
            answer  : Agent 的最终回答文本
            sources : 引用来源列表（文件名或 URL）
        """
        step = self._next_step()
        event = TraceEvent("final_answer", {
            "step": step,
            "answer": answer,
            "sources": sources or [],
        })
        self.events.append(event)
        _safe_print(f"[Answer] 步骤 {step} - 最终回答: {answer[:150]}...")

    def log_refuse(self, reason: str):
        """
        记录拒绝回答事件。

        当知识库中没有相关内容且不需要调用工具时，
        Agent 应该拒绝回答而不是编造答案（防止幻觉）。

        参数：
            reason : 拒绝回答的原因
        """
        step = self._next_step()
        event = TraceEvent("refuse", {
            "step": step,
            "reason": reason,
        })
        self.events.append(event)
        _safe_print(f"[Refuse] 步骤 {step} - 拒绝回答: {reason}")

    def log_error(self, error_message: str):
        """
        记录异常错误。

        参数：
            error_message : 错误描述
        """
        step = self._next_step()
        event = TraceEvent("error", {
            "step": step,
            "message": error_message,
        })
        self.events.append(event)
        _safe_print(f"[Error] 步骤 {step} - 错误: {error_message}")

    def print_summary(self):
        """
        打印完整的追踪报告摘要。

        这个报告展示了 Agent 每一步做了什么决策，
        是验收标准 4（全链路追踪）和标准 7（Prompt 迭代复盘）的关键依据。
        """
        print("\n" + "=" * 60)
        print("=== Agent 全链路追踪报告 ===")
        print("=" * 60)

        event_counts = {}
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        print(f"\n总事件数：{len(self.events)}")
        for event_type, count in event_counts.items():
            type_labels = {
                "reasoning": "[推理步骤]",
                "rag_search": "[RAG检索]",
                "tool_call": "[工具调用]",
                "final_answer": "[最终回答]",
                "refuse": "[拒绝回答]",
                "error": "[错误]",
            }
            label = type_labels.get(event_type, event_type)
            print(f"  {label}: {count} 次")

        print(f"\n详细事件列表：")
        for i, event in enumerate(self.events):
            print(f"  {i+1}. {event}")

        print("=" * 60)

    def to_dict_list(self) -> list:
        """将所有事件转为字典列表（方便导出 JSON）"""
        return [e.to_dict() for e in self.events]
