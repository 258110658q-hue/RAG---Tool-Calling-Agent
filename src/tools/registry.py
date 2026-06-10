"""
工具注册中心 (Tool Registry)
===============================

核心职责：
    作为所有工具的「中央登记处」，提供：
        1. 工具注册：将工具实例加入注册表
        2. Schema 导出：生成所有工具的 JSON Schema 列表（发给 LLM）
        3. 工具调用：根据 LLM 返回的 tool_name 和 arguments 执行对应工具

设计原则：
    1. 单一入口：Agent 只通过 Registry 与工具交互，不直接接触具体工具类。
    2. 可扩展：新增工具只需创建一个 BaseTool 子类，然后 register 进来。
    3. 错误隔离：一个工具执行出错不影响其他工具。

在整体架构中的位置：
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │  Agent   │ ←→  │   Registry   │ ←→  │   Tools      │
    │ (Phase 4)│     │   (本节)      │     │ (calculator, │
    └──────────┘     └──────────────┘     │  time_query) │
           │                │             └──────────────┘
           │                │
      "我需要计算"    execute("calculator", {"expression": "2+3"})
           │                │
           ▼                ▼
      LLM 决定调工具    Registry 找到 CalculatorTool，执行并返回结果
"""

from typing import List, Dict, Optional
from src.tools.base import BaseTool


class ToolRegistry:
    """
    工具注册中心。

    使用方法：
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(TimeTool())

        # 获取发给 LLM 的 Schema 列表
        schemas = registry.get_schemas()

        # 执行 LLM 指定的工具
        result = registry.execute("calculator", {"expression": "2+3"})
    """

    def __init__(self):
        """
        初始化一个空的注册表。

        _tools 字典的结构：
            {
                "calculator": <CalculatorTool 实例>,
                "time_query": <TimeTool 实例>,
            }
        """
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        注册一个工具。

        参数：
            tool : BaseTool 的子类实例

        如果工具名已存在，会打印警告但仍然覆盖（后注册的优先）。
        这是为了支持开发和调试阶段的快速迭代。
        """
        if tool.name in self._tools:
            print(f"[警告] 工具 '{tool.name}' 已注册，将被覆盖。")

        self._tools[tool.name] = tool
        print(f"[注册] 工具 '{tool.name}' 已就绪。")

    def get_schemas(self) -> List[dict]:
        """
        获取所有已注册工具的 JSON Schema 列表。

        这个列表直接作为 API 请求中 tools 参数的值。

        返回值示例：
        [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "计算数学表达式...",
                    "parameters": { ... }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "time_query",
                    "description": "查询当前的日期、时间...",
                    "parameters": { ... }
                }
            }
        ]
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        根据 LLM 的指令执行指定工具。

        参数：
            tool_name : LLM 返回的工具名称（如 "calculator"）
            arguments : LLM 返回的参数字典（如 {"expression": "2+3"}）

        返回值：
            工具执行结果的字符串（作为 tool role 的 content 返回给 LLM）

        异常处理：
            工具不存在 → 返回错误描述字符串（而不是抛出异常）
            工具执行失败 → 返回错误描述字符串

        为什么返回字符串而不是抛异常？
            因为即使工具执行失败，我们也需要把这个失败信息返回给 LLM，
            让 LLM 能够向用户解释"计算器说表达式有误，请修正"。
            如果抛异常，LLM 就收不到任何反馈，会一直重试同一个错误调用。
        """
        # --- 校验：工具是否存在 ---
        if tool_name not in self._tools:
            available = ", ".join(self._tools.keys())
            return (
                f"[系统错误] 未知工具 '{tool_name}'。"
                f"可用工具：{available}。"
            )

        tool = self._tools[tool_name]

        # --- 执行工具 ---
        try:
            # 使用 **arguments 将参数字典展开为关键字参数
            # 例如 execute(**{"expression": "2+3"}) 等价于 execute(expression="2+3")
            result = tool.execute(**arguments)
            return result

        except TypeError as error:
            # 参数不匹配（如缺少必填参数、参数名拼写错误）
            return (
                f"[工具错误] 参数不匹配：{error}。"
                f"工具 '{tool_name}' 期望的参数：{tool.parameters}"
            )

        except Exception as error:
            # 工具内部未预料的错误
            return (
                f"[工具错误] '{tool_name}' 执行失败："
                f"{type(error).__name__}: {error}"
            )

    def list_tools(self) -> List[str]:
        """
        列出所有已注册工具的名称。
        """
        return list(self._tools.keys())

    def __repr__(self):
        count = len(self._tools)
        names = ", ".join(self._tools.keys()) if self._tools else "(空)"
        return f"ToolRegistry({count} 个工具: {names})"
