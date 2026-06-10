"""
计算器工具 (Calculator Tool)
===============================

核心职责：
    安全地计算用户提供的数学表达式（加减乘除、括号等）。

安全设计（这是一个重要的工程考量）：
    Python 的 eval() 函数可以执行任意 Python 代码，如果直接 eval(用户输入)，
    用户可以输入 __import__('os').system('rm -rf /') 来执行系统命令。

    因此我们做了两层防护：
        1. 白名单正则过滤：只允许数字、运算符、括号、小数点、空格
        2. 沙箱命名空间：eval 时传入 {"__builtins__": {}}，禁用所有内置函数

    这样即使用户试图注入恶意代码，也会因包含非法字符而被正则过滤掉。
"""

import re
from src.tools.base import BaseTool


class CalculatorTool(BaseTool):
    """
    安全计算器工具。

    支持的运算：
        - 四则运算：+  -  *  /
        - 括号分组：( )
        - 小数：3.14
        - 幂运算：**  (如 2**10 = 1024)
    """

    # --- 工具元信息 ---
    name = "calculator"
    description = (
        "计算一个数学表达式的结果。"
        "支持加(+)、减(-)、乘(*)、除(/)、幂(**)、括号。"
        "示例表达式：'2 + 3 * 4'、'(15 + 25) / 2'、'2**10'。"
    )

    # --- JSON Schema 定义 ---
    # 这个 Schema 会发给 LLM，LLM 根据它来生成符合格式的调用参数。
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "要计算的数学表达式字符串。"
                    "只允许数字、运算符(+, -, *, /, **)、括号和小数点。"
                    "例如：'2 + 3'、'(10 - 2) * 5'、'2**8'。"
                ),
            },
        },
        "required": ["expression"],
    }

    # --- 允许的字符白名单 ---
    # 只有完全匹配这个正则表达式的输入才会被计算
    # 解释：^[\d\+\-\*\/\(\)\.\s]+$ 表示「从头到尾只包含数字、运算符、括号、小数点、空格」
    _SAFE_EXPRESSION_PATTERN = re.compile(r'^[\d\+\-\*\/\(\)\.\s]+$')

    def execute(self, expression: str = "") -> str:
        """
        安全地计算数学表达式。

        参数：
            expression : 数学表达式字符串（如 "2 + 3 * 4"）

        返回值：
            计算结果字符串（如 "14"），或错误描述
        """
        # --- 第 1 层防护：清理输入 ---
        # 去除首尾空白
        expression = expression.strip()

        if len(expression) == 0:
            return "[计算器错误] 表达式为空，请提供有效的数学表达式。"

        # --- 第 2 层防护：白名单正则过滤 ---
        if not self._SAFE_EXPRESSION_PATTERN.match(expression):
            return (
                f"[计算器错误] 表达式包含不允许的字符。"
                f"只允许：数字、运算符(+ - * /)、括号、小数点、空格。"
                f"收到的表达式：'{expression}'"
            )

        # --- 执行计算（沙箱环境） ---
        try:
            # eval 的三个参数：
            #   第 1 个：要计算的表达式字符串
            #   第 2 个：全局命名空间（空字典 = 不能访问任何全局变量）
            #   第 3 个：局部命名空间（空字典 = 不能访问任何内置函数）
            result = eval(expression, {"__builtins__": {}}, {})

            # 格式化结果：整数不显示小数点，浮点数保留 6 位有效数字
            if isinstance(result, (int, float)):
                if isinstance(result, float) and result == int(result):
                    # 对于像 2.0 这样的结果，显示为 2
                    result = int(result)
                elif isinstance(result, float):
                    # 浮点数保留适当精度
                    result = round(result, 10)

            return str(result)

        except ZeroDivisionError:
            return "[计算器错误] 除以零是不允许的。"
        except SyntaxError:
            return f"[计算器错误] 表达式语法不正确：'{expression}'"
        except Exception as error:
            return f"[计算器错误] 计算失败：{type(error).__name__}: {error}"
