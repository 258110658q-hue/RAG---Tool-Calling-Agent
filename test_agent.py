"""
Agent 全链路验证脚本
====================
运行方式：python test_agent.py

测试内容：
    在已索引的知识库基础上，让 Agent 回答多个不同类型的问题，
    覆盖：知识检索、工具调用、直接回答、拒绝回答四种场景。
"""

import sys
sys.path.insert(0, ".")

from src.agent.core import ReActAgent


def main():
    print("=" * 60)
    print("Agent 全链路验证：ReAct 循环 + RAG + 工具调用")
    print("=" * 60)

    # ============================================================
    # 第一步：初始化 Agent 并索引知识库
    # ============================================================
    print("\n[初始化] 创建 Agent 并加载知识库...")
    agent = ReActAgent()

    # 索引示例文档
    agent.index_documents([
        "data/向量数据库入门指南.md",
    ])

    # ============================================================
    # 第二步：多场景测试
    # ============================================================

    test_questions = [
        # 场景 1：纯知识库检索
        {
            "q": "什么是向量数据库？它和传统数据库有什么区别？",
            "expected_behavior": "应检索知识库并引用来源",
        },
        # 场景 2：需要计算工具
        {
            "q": "帮我算一下 (128 + 256) * 3 等于多少？",
            "expected_behavior": "应调用 calculator 工具",
        },
        # 场景 3：需要时间工具
        {
            "q": "今天是几月几号？星期几？",
            "expected_behavior": "应调用 time_query 工具",
        },
        # 场景 4：闲聊（不需要任何工具）
        {
            "q": "你好！请介绍一下你自己。",
            "expected_behavior": "应直接回答，不调用工具",
        },
        # 场景 5：知识库中不存在的内容（触发拒绝回答）
        {
            "q": "量子计算机的工作原理是什么？请详细解释。",
            "expected_behavior": "应触发拒绝回答或明确表示知识库中无此信息",
        },
    ]

    for i, test in enumerate(test_questions):
        print(f"\n\n{'#'*60}")
        print(f"# 测试 {i+1}/{len(test_questions)}")
        print(f"# 问题：{test['q']}")
        print(f"# 预期行为：{test['expected_behavior']}")
        print(f"{'#'*60}")

        result = agent.run(test["q"])

        print(f"\n{'─'*60}")
        print(f"📝 最终回答：")
        print(f"{result['answer']}")
        print(f"\n📎 引用来源：{result['sources'] if result['sources'] else '(无)'}")
        print(f"🔄 使用迭代：{result['iterations']} 轮")

        # 打印追踪报告
        result["trace"].print_summary()

    print("\n" + "=" * 60)
    print("✅ Agent 全链路验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
