"""
SSE 底层原理演示脚本
=====================
目的：让你亲眼看到 OpenAI SDK 在流式模式下，从服务器收到的
     每一个「原始数据块 (chunk)」的完整结构。

运行方式：python demo_sse_raw.py

学完这个脚本，你会理解：
    1. chunk 到底是什么
    2. delta 和 message 的区别
    3. 为什么我们 chat_stream() 里用 delta.content 而不是 message.content
"""

import sys
sys.path.insert(0, ".")

from src.llm.client import LLMClient
from src.config import config


# 创建一个不经过我们封装的「裸」OpenAI 客户端，
# 直接调用 SDK 的流式方法，打印每一个 chunk 的全部字段。
from openai import OpenAI

raw_client = OpenAI(
    api_key=config.llm_api_key,
    base_url=config.llm_base_url,
)

messages = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "用一句话解释什么是 token。"},
]

print("=" * 60)
print("SSE 原始 Chunk 结构演示")
print("=" * 60)
print()
print("下面每个「---------- chunk #N ----------」")
print("就是服务器通过 SSE 协议推送的一个数据块。")
print()

# 调用 SDK 的流式方法
stream = raw_client.chat.completions.create(
    model=config.llm_model,
    messages=messages,
    temperature=0.3,
    stream=True,  # ← 关键！开启流式
)

chunk_count = 0  # 计数器：我们总共收到多少个 chunk

for chunk in stream:
    chunk_count += 1

    # 每个 chunk 是一个 ChatCompletionChunk 对象
    # 它的核心结构是 chunk.choices[0].delta
    choice = chunk.choices[0]
    delta = choice.delta

    # --- 显示这个 chunk 的关键信息 ---
    print(f"---------- chunk #{chunk_count} ----------")

    # finish_reason：为什么结束？None = 还在生成, "stop" = 正常结束
    print(f"  finish_reason : {choice.finish_reason}")

    # delta.content：这个 chunk 携带的新增文本（如果是 Function Calling 则为 None）
    print(f"  delta.content : {repr(delta.content)}")

    # delta.role：只有第一个 chunk 会返回（如 "assistant"），后面都是 None
    print(f"  delta.role    : {repr(delta.role)}")

    # delta.tool_calls：Function Calling 的工具调用增量（Phase 3 详细讲解）
    print(f"  delta.tool_calls : {repr(delta.tool_calls)}")

    print()

print("=" * 60)
print(f"总共收到 {chunk_count} 个 chunk")
print()
print("关键观察：")
print("  1. 除了第 1 个 chunk 的 delta.role='assistant'，其余都是 None")
print("  2. 除了最后一个 chunk 的 finish_reason='stop'，其余都是 None")
print("  3. 文本内容分散在多个 chunk 的 delta.content 中，需要拼接")
print("=" * 60)
