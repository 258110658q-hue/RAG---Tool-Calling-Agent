"""
工具基类模块 (Tool Base)
==========================

核心职责：
    定义所有工具必须遵守的「接口契约」——每个工具都必须提供：
        1. 一个 JSON Schema 描述（告诉 LLM 怎么调用这个工具）
        2. 一个 execute 方法（真正执行工具逻辑）

设计原则：
    1. 接口优先：先定契约，再写实现。所有工具遵循相同的调用规范。
    2. 自描述：每个工具自带 JSON Schema，LLM 能自动理解如何调用。
    3. 安全隔离：工具的 execute 方法独立运行，互不影响。

关于「抽象基类」vs「简单继承」的选择：
    这里我们不使用 Python 的 ABC（Abstract Base Class），
    而是用简单的「约定优于配置」——子类覆盖 name、description、parameters
    和 execute 方法即可。这对新手更友好，且足够覆盖我们的需求。

关于 JSON Schema（你需要深刻理解这个概念）：
    JSON Schema 是一种描述 JSON 数据格式的「元语言」。
    当我们把工具的 JSON Schema 传给 LLM 时，我们本质上是在说：
    「嘿，这个工具接受以下格式的参数，请你严格按这个格式生成调用。」

    例如 Calculator 的 Schema：
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式"
            }
        },
        "required": ["expression"]
    }

    LLM 看到这个 Schema 后，会生成：
    {"expression": "2 + 3 * 4"}

    而不是胡乱生成：
    {"formula": "2+3*4", "precision": 5}  ← Schema 中没有这些字段，LLM 不会乱编
"""


class BaseTool:
    """
    工具的基类（接口契约）。

    每个具体工具继承这个类，并实现：
        - name        : 工具的唯一名称（如 "calculator"）
        - description : 工具的功能描述（LLM 用它来判断何时调用）
        - parameters  : 参数的 JSON Schema 定义
        - execute()   : 执行工具逻辑的方法

    使用示例（子类）：
        class CalculatorTool(BaseTool):
            name = "calculator"
            description = "计算数学表达式的结果"
            parameters = { ... }
            def execute(self, expression): ...
    """

    # 子类必须覆盖这些属性
    name: str = ""          # 工具名称（英文、小写、下划线分隔）
    description: str = ""   # 工具描述（给 LLM 看的，中文也可以）

    # 参数的 JSON Schema（子类覆盖）——只定义 "properties" 和 "required"
    # 完整的 Function Calling Schema 由 to_openai_schema() 方法拼装
    parameters: dict = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, **kwargs):
        """
        执行工具逻辑。

        子类必须覆盖这个方法。
        **kwargs 接收 LLM 传入的参数（字段名与 parameters 中的 properties 对应）。

        返回值：
            字符串 —— 工具的执行结果（会直接作为 tool role 的 content 返回给 LLM）
        """
        raise NotImplementedError(
            f"工具 '{self.name}' 未实现 execute() 方法。"
        )

    def to_openai_schema(self) -> dict:
        """
        生成 OpenAI / DeepSeek Function Calling 格式的工具描述。

        这是协议层的 JSON 结构——LLM 通过它来理解工具的用途和参数格式。
        返回的字典直接放入 API 请求的 tools 数组中。

        返回值示例：
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "计算数学表达式",
                "parameters": { ... JSON Schema ... }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
