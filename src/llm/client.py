"""
LLM 客户端模块 (LLM Client)
=============================

核心职责：
    封装与大语言模型（LLM）的全部通信细节，向业务代码提供
    两个干净、简单的方法：
        1. chat()        —— 普通对话（一次性返回完整回复）
        2. chat_stream() —— 流式对话（逐 token 返回，像打字机效果）

设计原则：
    1. 关注点分离：业务代码不需要知道 HTTP 请求 / API Key / 重试逻辑。
    2. 统一异常处理：所有网络层面的错误在这里集中处理，业务代码只关心业务。
    3. 可替换性：如果未来换模型供应商，只需改这一个文件。

关于"重试"的底层概念（你真的需要懂）：
    当我们的程序向 LLM 服务器发请求时，可能遇到三类错误：
    ┌──────────────┬──────────────────────────────────┐
    │ 错误类型      │ 典型原因                          │
    ├──────────────┼──────────────────────────────────┤
    │ 网络超时      │ 服务器太忙，响应太慢（可重试）       │
    │ 限流 (429)    │ 请求频率超过限制，等几秒再试（可重试）│
    │ 认证失败 (401)│ API Key 错了 —— 重试 100 次也没用   │
    └──────────────┴──────────────────────────────────┘

    因此我们采用「指数退避」策略：
        第 1 次失败后等待 1 秒，第 2 次等 2 秒，第 3 次等 4 秒…
        但认证错误（401）立即放弃，不浪费时间和配额。
"""

import time  # 用于重试时的等待（sleep）
from openai import OpenAI  # OpenAI 官方 SDK —— 兼容 DeepSeek 等供应商
from src.config import config  # 我们刚才写的配置模块


# ============================================================
# LLM 客户端类
# ============================================================

class LLMClient:
    """
    大模型对话客户端。

    这是整个项目与大模型通信的「唯一出入口」。
    所有模块（Agent、RAG、Tools）都通过这个类与 LLM 交互。
    """

    def __init__(self):
        """
        初始化客户端。

        这里做的事情：
            1. 从 config 中读取 API Key、Base URL 等配置
            2. 创建一个 OpenAI SDK 的客户端实例
            3. 把这个实例保存到 self._client 中，后续所有方法都用它来发请求

        注意：变量名前面的下划线 _client 是 Python 的约定，
        意思是「这是一个内部属性，外部调用者不应该直接访问它」。
        """
        # 从统一配置模块读取所有参数
        api_key = config.llm_api_key
        base_url = config.llm_base_url

        # 安全检查：如果 API Key 为空，立即报错并给出明确提示
        # 这样比等到发请求时才报 401 错误更容易排查问题
        if not api_key:
            raise ValueError(
                "LLM API Key 未设置！请检查 .env 文件中的 LLM_API_KEY 值。\n"
                "如果你还没有 .env 文件，请复制 .env.example 并填入真实的 Key。"
            )

        # 创建 OpenAI SDK 客户端实例
        # OpenAI() 构造函数接收 api_key 和 base_url 两个核心参数：
        #   - api_key：每次请求时放在 HTTP Header 中的 Authorization: Bearer <key>
        #   - base_url：API 服务器地址，默认是 https://api.openai.com/v1
        #     你改成了 https://api.deepseek.com/v1，SDK 就会把请求发到 DeepSeek
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 保存常用参数，方便后续方法使用
        self._model = config.llm_model
        self._max_retries = config.max_retries
        self._timeout = config.request_timeout

    # ============================================================
    # 公开方法 1：普通对话（非流式）
    # ============================================================

    def chat(
        self,
        messages,          # list[dict]：对话历史，格式为 [{"role": "user", "content": "你好"}]
        temperature=0.7,   # float：生成随机性 (0=确定性, 1=创造性)
        tools=None,        # list[dict] | None：Function Calling 的工具定义（Phase 3 会用到）
    ):
        """
        发送一次对话请求，等待 LLM 完整回复后返回。

        参数说明：
            messages：
                对话消息列表，每一条是一个字典，必须包含 role 和 content 两个字段。
                role 有三种：'system'（系统指令）、'user'（用户消息）、'assistant'（模型回复）。
                示例：
                    [
                        {"role": "system", "content": "你是一个有用的助手。"},
                        {"role": "user", "content": "今天天气怎么样？"},
                    ]

            temperature：
                控制输出的随机程度，范围 0.0 ~ 2.0。
                0.0 = 几乎每次返回相同的答案（适合数学/RAG 检索）;
                1.0 = 有创造性（适合写作）

            tools：
                Function Calling 的工具定义列表。None 表示不使用工具。
                这个参数将在 Phase 3 中深入讲解。

        返回值：
            一个字符串，即 LLM 的完整回复内容。

        内部流程：
            1. 调用 _invoke_with_retry() 发请求（内置重试逻辑）
            2. 从响应对象中提取纯文本回复
            3. 返回给调用者
        """
        # 调用内部方法（带重试）
        response = self._invoke_with_retry(
            messages=messages,
            temperature=temperature,
            tools=tools,
        )

        # 从 OpenAI 的响应对象中提取文本内容
        # response.choices 是一个列表，通常只有一个元素（除非设置了 n>1）
        # choices[0].message.content 就是模型回复的文本
        choice = response.choices[0]
        message = choice.message

        # 如果模型返回了内容，就返回它；否则返回空字符串
        if message.content is not None:
            return message.content
        else:
            # 这种情况极少发生（通常只有 Function Calling 时 content 才为 None）
            return ""

    # ============================================================
    # 公开方法 2：流式对话（逐 token 返回）
    # ============================================================

    def chat_stream(
        self,
        messages,          # list[dict]：对话历史
        temperature=0.7,   # float：生成随机性
        tools=None,        # list[dict] | None：工具定义
    ):
        """
        以流式（Streaming）方式发送对话请求，逐 token 返回回复。

        这是下一步骤 1.4 要深入讲解的核心功能。
        现在先把「普通对话」跑通，建立信心后我们再进入流式。

        参数说明：与 chat() 方法完全一致。

        返回值：
            一个生成器（Generator），调用者通过 for 循环逐个获取 token。

        「生成器」是什么？
            你可以把它理解为一个「可以暂停和恢复的函数」。
            每次执行到 yield 时，函数暂停，把值返回给调用者；
            下次调用者请求下一个值时，函数从暂停处继续执行。
            这样不会一次性占用大量内存，特别适合处理大数据流。
        """
        # 调用内部重试方法，传入 stream=True 开启流式
        stream = self._invoke_with_retry(
            messages=messages,
            temperature=temperature,
            tools=tools,
            stream=True,  # 关键参数：告诉 API 以流式方式返回
        )

        # 逐块处理服务器发回的 SSE 事件
        for chunk in stream:
            # 每个 chunk 包含 choices[0].delta.content
            # delta 是「增量」，即本次新增的几个 token（不是从头累积）
            choice = chunk.choices[0]
            delta = choice.delta

            # 如果 delta.content 不为空，就 yield 出去
            if delta.content is not None:
                yield delta.content

    # ============================================================
    # 内部方法：带重试逻辑的请求
    # ============================================================

    def _invoke_with_retry(
        self,
        messages,
        temperature=0.7,
        tools=None,
        stream=False,
    ):
        """
        发送 API 请求，并在遇到可恢复错误时自动重试。

        这是整个 LLMClient 的「心脏」——所有对外公开的方法
        （chat / chat_stream）最终都通过它来与服务器通信。

        重试策略（指数退避）：
            第 1 次失败 → 等待 1 秒后重试
            第 2 次失败 → 等待 2 秒后重试
            第 3 次失败 → 等待 4 秒后重试

            但以下错误立即放弃（不重试）：
            - 认证错误（401）：API Key 错了，重试无意义
            - 权限错误（403）：没有访问权限

        参数说明：
            stream：
                False = 等完整结果一次性返回
                True  = 以 SSE 流逐个返回 token

        返回值：
            stream=False 时返回一个完整的 ChatCompletion 对象
            stream=True  时返回一个可迭代的 Stream 对象

        异常处理：
            如果重试耗尽仍然失败，抛出包含详细错误信息的 RuntimeError。
        """
        last_error = None  # 记录最后一次失败的错误信息

        # 重试循环：最多尝试 self._max_retries 次
        for attempt in range(1, self._max_retries + 1):
            try:
                # --- 核心 API 调用 ---
                # 这是整个方法唯一真正发请求的地方。
                # self._client.chat.completions.create() 会：
                #   1. 把参数序列化为 JSON
                #   2. 通过 HTTPS POST 发送到 base_url/chat/completions
                #   3. 等待服务器返回
                #   4. 把 JSON 响应解析为 Python 对象
                response = self._client.chat.completions.create(
                    model=self._model,        # 模型名称，如 deepseek-coder
                    messages=messages,         # 对话历史
                    temperature=temperature,   # 生成随机度
                    tools=tools,               # 工具定义 (Phase 3)
                    stream=stream,             # 是否流式返回
                    timeout=self._timeout,     # 超时时间（秒）
                )
                # 成功！返回响应对象
                return response

            except Exception as error:
                # 记录这次失败
                last_error = error
                error_type = type(error).__name__  # 获取错误类型名称
                error_message = str(error)          # 获取错误描述

                # --- 判断是否值得重试 ---
                # 某些错误是「永久性」的，重试不会改变结果
                if self._is_non_retryable(error):
                    # 不可恢复的错误：立即抛出，不浪费时间和配额
                    raise RuntimeError(
                        f"LLM API 调用失败（不可恢复的错误，未重试）：\n"
                        f"  错误类型：{error_type}\n"
                        f"  错误详情：{error_message}"
                    )

                # --- 判断是否还有重试机会 ---
                if attempt < self._max_retries:
                    # 还有重试机会：等待后继续
                    wait_seconds = 2 ** (attempt - 1)  # 指数退避：1, 2, 4, 8...
                    print(
                        f"[重试] 第 {attempt} 次尝试失败（{error_type}），"
                        f"{wait_seconds} 秒后进行第 {attempt + 1} 次重试..."
                    )
                    time.sleep(wait_seconds)
                else:
                    # 重试次数已耗尽
                    raise RuntimeError(
                        f"LLM API 调用失败（已重试 {self._max_retries} 次，全部失败）：\n"
                        f"  最后一次错误类型：{error_type}\n"
                        f"  最后一次错误详情：{error_message}"
                    )

    def _is_non_retryable(self, error) -> bool:
        """
        判断一个错误是否「不可重试」。

        返回 True 表示这个错误重试没有意义（如 API Key 错误），
        应该立即放弃而不是等待重试。

        返回 False 表示这个错误可能是临时性的（如网络超时），
        可以等待后重试。
        """
        error_str = str(error).lower()

        # 401: 认证失败 —— API Key 无效或过期
        if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
            return True

        # 403: 权限不足 —— 没有访问该模型或资源的权限
        if "403" in error_str or "forbidden" in error_str or "permission" in error_str:
            return True

        # 其他错误（如网络超时、限流 429、服务器错误 500 等）都认为可重试
        return False
