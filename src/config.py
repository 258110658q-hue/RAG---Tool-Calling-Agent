"""
配置管理模块 (Configuration Manager)
======================================

核心职责：
    从 .env 文件中安全地读取所有配置项，并向项目的其他模块
    提供一个统一、只读的配置访问入口。

设计原则：
    1. 单一数据源 (Single Source of Truth)：所有配置只从这一个模块获取。
    2. 敏感信息隔离：API Key 等密钥存在 .env 文件中，不进入版本控制。
    3. 默认值保护：每一项配置都有安全的默认值，程序不会因为漏配而崩溃。

使用方法：
    from src.config import config
    print(config.llm_api_key)  # 获取 LLM API 密钥
"""

import os
from dotenv import load_dotenv


# ============================================================
# 第一步：加载 .env 文件
# ============================================================
# load_dotenv() 会找到项目根目录下的 .env 文件，
# 把它里面的 KEY=VALUE 逐行解析，注入到系统环境变量中。
# 之后就可以用 os.getenv("KEY") 来读取对应的值了。
# ============================================================

load_dotenv()


# ============================================================
# 第二步：定义配置类
# ============================================================
# 为什么用类 (Class) 而不是散落的函数 / 变量？
#
# 答：类可以把相关的配置「打包」在一起，形成清晰的命名空间。
#     比如 config.llm_api_key 比 config_get_llm_api_key() 更直观。
#     而且类天然支持「只读属性」，防止代码在运行时意外修改配置。
#
# 每一个属性（用 @property 装饰的方法）代表一项配置。
# @property 的作用：把这个方法变成一个「看起来像普通属性」的只读字段，
# 外部访问时不需要写括号：config.llm_model（而不是 config.llm_model()）。
# ============================================================

class Config:
    """
    项目全局配置类。

    每新增一个配置项，就在这里添加一个对应的 @property 方法。
    这样所有模块都能通过同一个入口获取配置，避免散落各处、难以维护。
    """

    # --- LLM 相关配置 ---

    @property
    def llm_api_key(self) -> str:
        """
        LLM API 密钥。

        从环境变量 LLM_API_KEY 中读取。
        如果 .env 文件中没有设置，则返回空字符串（后续调用会报错提示）。
        """
        return os.getenv("LLM_API_KEY", "")

    @property
    def llm_base_url(self) -> str:
        """
        LLM API 的服务地址。

        如果你用的是 OpenAI 官方服务，不需要改这个值。
        如果你用的是国内中转代理（如 APIHub、OpenRouter 等），
        把 .env 中的 LLM_BASE_URL 改为代理服务商提供的地址即可。

        默认值：OpenAI 官方地址
        """
        return os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    @property
    def llm_model(self) -> str:
        """
        默认使用的模型名称。

        - gpt-4o：OpenAI 当前的主力模型（性价比高）
        - gpt-4-turbo：上一代旗舰（稍慢但稳定）
        - deepseek-chat：DeepSeek 的对话模型（如果你走国内代理）

        可以在 .env 中修改 LLM_MODEL 来切换。
        """
        return os.getenv("LLM_MODEL", "gpt-4o")

    # --- Embedding 相关配置 ---

    @property
    def embedding_api_key(self) -> str:
        """
        Embedding 专用的 API Key。

        如果你的 LLM 对话和 Embedding 用的是不同服务商
        （比如：对话用 DeepSeek，Embedding 用通义千问），
        就在这里单独设置 Embedding 的 Key。

        如果为空，默认复用 llm_api_key（同一服务商的情况）。
        """
        return os.getenv("EMBEDDING_API_KEY", "") or self.llm_api_key

    @property
    def embedding_model(self) -> str:
        """
        Embedding 模型名称。

        Embedding 模型用于将文本转换为向量（数值数组）。
        它和对话模型（LLM）是两种不同的模型，各司其职：
        - LLM：接收文本，生成文本（聊天、推理、生成）
        - Embedding 模型：接收文本，输出向量（检索、聚类、相似度计算）

        常用的 Embedding 模型：
        - text-embedding-3-small：OpenAI 最新轻量模型，1536 维
        - text-embedding-ada-002：OpenAI 上一代模型，1536 维
        - 如果你的 API 供应商（如 DeepSeek）提供 Embedding 服务，
          在这里填写对应的模型名
        """
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    @property
    def embedding_base_url(self) -> str:
        """
        Embedding API 的服务地址。

        有些供应商的 Embedding 端点和对话端点是分开的。
        如果为空，默认复用 llm_base_url（大多数情况适用）。
        """
        return os.getenv("EMBEDDING_BASE_URL", "")

    # --- 应用级配置 ---

    @property
    def max_retries(self) -> int:
        """
        API 调用失败时的最大重试次数。

        网络请求可能因为各种原因失败（超时、限流、服务器临时错误），
        设置重试次数可以提升程序的鲁棒性（健壮性）。

        注意：int(os.getenv(...)) 做了类型转换，
        因为环境变量默认都是字符串。
        """
        raw_value = os.getenv("MAX_RETRIES", "3")
        return int(raw_value)

    @property
    def request_timeout(self) -> int:
        """
        API 请求的超时时间（单位：秒）。

        如果 LLM 在这个时间内没有返回结果，就主动断开连接，
        避免程序无限等待。
        """
        raw_value = os.getenv("REQUEST_TIMEOUT", "60")
        return int(raw_value)


# ============================================================
# 第三步：创建全局唯一的配置实例
# ============================================================
# 这个名字写作全小写的 config（区别于大驼峰的类名 Config），
# 其他模块导入时写：
#
#   from src.config import config
#
# 这样整个项目只用这一个实例，确保配置的一致性。
# ============================================================

config = Config()
