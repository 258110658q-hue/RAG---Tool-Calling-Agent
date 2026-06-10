"""
工具系统验证脚本
================
运行方式：python test_tools.py

测试内容：
    1. 单元测试：直接调用工具验证功能
    2. Schema 验证：检查 JSON Schema 格式
    3. Registry 验证：测试注册中心的增删查
    4. Function Calling 集成测试：让 LLM 实际使用工具
"""

import sys
sys.path.insert(0, ".")

from src.tools.calculator import CalculatorTool
from src.tools.time_tool import TimeTool
from src.tools.registry import ToolRegistry
from src.llm.client import LLMClient


def test_calculator_direct():
    """测试 1：直接调用计算器"""
    print("=" * 50)
    print("测试 1：计算器工具（直接调用）")
    print("=" * 50)

    calc = CalculatorTool()

    # 正常用例
    cases = [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("3 * 7", "21"),
        ("15 / 3", "5.0"),
        ("2 ** 8", "256"),
        ("(10 + 2) * 3", "36"),
    ]

    for expression, expected in cases:
        result = calc.execute(expression=expression)
        status = "✅" if expected in result else "❌"
        print(f"  {status} '{expression}' = {result}")

    # 异常用例
    print("\n  异常处理测试：")
    print(f"  除以零：{calc.execute(expression='1/0')}")
    print(f"  非法字符：{calc.execute(expression='__import__(\"os\")')}")
    print(f"  空表达式：{calc.execute(expression='')}")


def test_time_direct():
    """测试 2：直接调用时间工具"""
    print("\n" + "=" * 50)
    print("测试 2：时间工具（直接调用）")
    print("=" * 50)

    time_tool = TimeTool()

    actions = ["now", "date", "time", "weekday", "timestamp"]
    for action in actions:
        result = time_tool.execute(action=action, timezone_offset="+8")
        print(f"  {action:12} → {result}")


def test_registry():
    """测试 3：工具注册中心"""
    print("\n" + "=" * 50)
    print("测试 3：工具注册中心")
    print("=" * 50)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())

    print(f"\n  已注册工具：{registry.list_tools()}")
    print(f"  Registry 状态：{registry}")

    # 通过 Registry 执行工具
    result = registry.execute("calculator", {"expression": "100 + 200"})
    print(f"  registry.execute('calculator', {{'expression': '100+200'}}) = {result}")

    # 测试未知工具
    result = registry.execute("unknown_tool", {})
    print(f"  调用不存在工具：{result[:80]}...")

    # 获取 Schema
    schemas = registry.get_schemas()
    print(f"\n  生成的 JSON Schema 数量：{len(schemas)}")
    for schema in schemas:
        func = schema["function"]
        print(f"    - {func['name']}: {func['description'][:50]}...")


def test_function_calling_with_llm():
    """
    测试 4：Function Calling 集成测试

    这是 Phase 3 最重要的测试——让 LLM 在真实对话中决定是否调用工具。
    我们会看到 LLM 如何返回 tool_calls，以及我们的代码如何处理它们。
    """
    print("\n" + "=" * 50)
    print("测试 4：Function Calling 集成测试")
    print("=" * 50)

    # 初始化
    client = LLMClient()
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())

    # 测试用的对话——故意让模型需要用到工具
    test_cases = [
        {
            "label": "需要计算器",
            "messages": [
                {"role": "system", "content": "你是一个助手。当需要计算时，使用 calculator 工具；当需要知道时间时，使用 time_query 工具。如果不需要工具就直接回答。"},
                {"role": "user", "content": "帮我算一下 156 * 23 + 789 等于多少？"},
            ],
        },
        {
            "label": "需要时间查询",
            "messages": [
                {"role": "system", "content": "你是一个助手。当需要计算时，使用 calculator 工具；当需要知道时间时，使用 time_query 工具。"},
                {"role": "user", "content": "今天是星期几？"},
            ],
        },
        {
            "label": "不需要工具",
            "messages": [
                {"role": "system", "content": "你是一个助手。当需要计算时，使用 calculator 工具；当需要知道时间时，使用 time_query 工具。如果不需要工具就直接回答。"},
                {"role": "user", "content": "什么是Python？用一句话回答。"},
            ],
        },
    ]

    for case in test_cases:
        print(f"\n{'─' * 50}")
        print(f"  场景：{case['label']}")
        print(f"  用户：{case['messages'][-1]['content']}")

        # --- 第一次调用 LLM ---
        # 传入 tools Schema，让 LLM 决定是直接回答还是调用工具
        tools = registry.get_schemas()
        response = client._invoke_with_retry(
            messages=case["messages"],
            temperature=0.0,
            tools=tools,
        )

        # 检查 LLM 的选择
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            # LLM 决定调用工具！
            print(f"  → LLM 决定调用工具：")

            # 构建完整的 assistant 消息（包含 tool_calls）
            # 这是协议要求：assistant 的 tool_calls 消息必须完整保留
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

            # 把 assistant 的 tool_calls 消息加入对话历史
            case["messages"].append(assistant_message)

            # 逐个执行每个工具调用
            for tc in message.tool_calls:
                tool_name = tc.function.name
                import json
                arguments = json.loads(tc.function.arguments)

                print(f"    - 工具：{tool_name}")
                print(f"    - 参数：{arguments}")

                # 执行工具
                result = registry.execute(tool_name, arguments)
                print(f"    - 结果：{result}")

                # 把工具结果加入对话历史（tool role）
                case["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # --- 第二次调用 LLM（携带工具结果）---
            # 这次 LLM 会综合工具结果，生成面向用户的回答
            final_response = client._invoke_with_retry(
                messages=case["messages"],
                temperature=0.0,
            )
            final_text = final_response.choices[0].message.content
            print(f"  → LLM 最终回答：{final_text}")

        else:
            # LLM 决定直接回答
            print(f"  → LLM 直接回答（未调用工具）：{message.content[:150]}...")


if __name__ == "__main__":
    test_calculator_direct()
    test_time_direct()
    test_registry()
    test_function_calling_with_llm()

    print("\n" + "=" * 50)
    print("✅ Phase 3 工具系统验证完成。")
    print("=" * 50)
