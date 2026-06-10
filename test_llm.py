"""
LLM 客户端验证脚本
==================
运行方式：在项目根目录下执行
    python test_llm.py

测试内容：
    1. 普通对话（非流式）—— 发送一条消息，等待完整回复
    2. 流式对话           —— 发送一条消息，逐 token 输出
"""

import sys
sys.path.insert(0, ".")

from src.llm.client import LLMClient


def test_chat():
    """
    测试 1：普通对话（非流式）

    发送一个简单的问题，验证 LLM 能否正常返回完整回复。
    """
    print("=" * 50)
    print("测试 1：普通对话（非流式）")
    print("=" * 50)

    # 第一步：创建客户端实例
    # __init__ 会读取配置、检查 API Key、初始化 OpenAI SDK
    client = LLMClient()

    # 第二步：构造对话消息
    # messages 是一个列表，每个元素是一个字典
    # system 角色 —— 设定 AI 的行为模式
    # user   角色 —— 用户的具体问题
    messages = [
        {"role": "system", "content": "你是一个简洁的助手，回答问题时不超过三句话。"},
        {"role": "user", "content": "什么是向量数据库？用最通俗的话解释。"},
    ]

    # 第三步：调用 chat() 方法
    # temperature=0.3 —— 较低的随机度，让回答更稳定
    print("\n[用户问题] 什么是向量数据库？\n")
    print("[模型回复] ", end="", flush=True)

    reply = client.chat(messages=messages, temperature=0.3)

    print(reply)
    print()
    return client  # 返回客户端，供下一个测试使用


def test_chat_stream(client):
    """
    测试 2：流式对话

    发送一个问题，观察逐 token 返回的效果（打字机效果）。
    """
    print("=" * 50)
    print("测试 2：流式对话")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "你是一个简洁的助手，回答不超过三句话。"},
        {"role": "user", "content": "解释一下什么是 Embedding？"},
    ]

    print("\n[用户问题] 什么是 Embedding？\n")
    print("[模型回复（流式）] ", end="", flush=True)

    # chat_stream() 返回一个生成器，用 for 循环逐个获取 token
    for token in client.chat_stream(messages=messages, temperature=0.3):
        # 每收到一个 token 就立即打印，不换行
        print(token, end="", flush=True)

    print("\n")
    print("✅ 流式对话测试完成。")
    print()


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    try:
        # 先运行测试 1
        client = test_chat()

        # 再运行测试 2
        test_chat_stream(client)

        print("=" * 50)
        print("✅ 所有测试通过！LLM 客户端封装成功。")
        print("=" * 50)

    except ValueError as e:
        # 捕获配置错误（如 API Key 未设置）
        print(f"\n❌ 配置错误：{e}")

    except RuntimeError as e:
        # 捕获 API 调用错误（如网络问题、认证失败）
        print(f"\n❌ API 调用错误：{e}")

    except Exception as e:
        # 捕获其他未预料的错误
        print(f"\n❌ 未知错误：{type(e).__name__}: {e}")
