"""
时间查询工具 (Time Query Tool)
=================================

核心职责：
    提供日期、时间、星期等时间信息的查询能力。

为什么 Agent 需要时间工具？
    LLM 的训练数据有截止日期，它不知道「现在」是什么时间。
    你把这个问题交给 LLM：
        "今天是几号？"
    LLM 如果不借助工具，只能「猜测」或者告诉你它不知道。
    有了时间工具，Agent 就能获取真实的当前时间。

设计思路：
    这个工具用「操作码 (action)」模式——一个工具支持多种操作，
    通过 action 参数来区分。这样做的好处：
        - 减少工具数量：不需要 get_date、get_time、get_weekday 三个独立工具
        - 语义集中：LLM 看到 "time_query" 就知道和时间相关
        - JSON Schema 中 pros 的 enum 字段告诉 LLM 可选值，减少歧义
"""

from datetime import datetime, timezone, timedelta
from src.tools.base import BaseTool


class TimeTool(BaseTool):
    """
    时间查询工具。

    支持的操作：
        - now：当前日期和时间
        - date：仅日期
        - time：仅时间
        - weekday：今天是星期几
        - timestamp：当前 Unix 时间戳
    """

    # --- 工具元信息 ---
    name = "time_query"
    description = (
        "查询当前的日期、时间、星期等信息。"
        "当用户询问与时间相关的问题时使用此工具。"
    )

    # --- JSON Schema 定义 ---
    # 注意 action 字段使用了 "enum"——这告诉 LLM 只能从这 5 个值中选择
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["now", "date", "time", "weekday", "timestamp"],
                "description": (
                    "要查询的时间信息类型：\n"
                    "  - 'now': 完整的当前日期和时间\n"
                    "  - 'date': 仅当前日期（年-月-日）\n"
                    "  - 'time': 仅当前时间（时:分:秒）\n"
                    "  - 'weekday': 今天是星期几\n"
                    "  - 'timestamp': 当前 Unix 时间戳（秒）"
                ),
            },
            "timezone_offset": {
                "type": "string",
                "description": (
                    "时区偏移量，格式如 '+8'（东八区/北京时间）、'-5'（美国东部）。"
                    "如果不传这个参数，默认使用 UTC 时间。"
                    "中国用户查询时，通常应传 '+8'。"
                ),
            },
        },
        "required": ["action"],
    }

    # 星期几的中文映射
    _WEEKDAY_NAMES = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

    def execute(self, action: str = "", timezone_offset: str = "+8") -> str:
        """
        执行时间查询。

        参数：
            action          : 操作类型（now/date/time/weekday/timestamp）
            timezone_offset : 时区偏移字符串（如 '+8'）

        返回值：
            格式化的时间信息字符串
        """
        # --- 解析时区 ---
        try:
            # 将 '+8' 或 '-5' 这样的字符串转为整数偏移量
            offset_hours = int(timezone_offset) if timezone_offset else 0
            # 创建一个带有时区偏移的时间对象
            tz = timezone(timedelta(hours=offset_hours))
        except (ValueError, TypeError):
            # 时区解析失败时使用 UTC
            tz = timezone.utc
            offset_hours = 0

        # 获取指定时区的当前时间
        now = datetime.now(tz)

        # --- 根据操作类型生成结果 ---
        action = action.strip().lower()

        if action == "now":
            return now.strftime("%Y年%m月%d日 %H:%M:%S") + f" (UTC{timezone_offset})"

        elif action == "date":
            return now.strftime("%Y年%m月%d日")

        elif action == "time":
            return now.strftime("%H:%M:%S") + f" (UTC{timezone_offset})"

        elif action == "weekday":
            # Python 的 weekday() 返回 0=周一, 6=周日
            weekday_index = now.weekday()
            weekday_name = self._WEEKDAY_NAMES[weekday_index]
            return f"{now.strftime('%Y年%m月%d日')} 是 {weekday_name}"

        elif action == "timestamp":
            # Unix 时间戳 = 从 1970-01-01 00:00:00 UTC 到现在的秒数
            ts = int(now.timestamp())
            return f"当前 Unix 时间戳：{ts}"

        else:
            # 未知操作：返回支持的操作列表
            supported = ", ".join([
                f"'{a}'" for a in
                self.parameters["properties"]["action"]["enum"]
            ])
            return f"[时间工具错误] 不支持的操作 '{action}'。支持的操作：{supported}。"
