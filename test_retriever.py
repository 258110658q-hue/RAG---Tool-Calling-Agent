"""
RAG 全链路验证脚本
==================
运行方式：python test_retriever.py

测试内容：
    1. 解析 Markdown 文档
    2. 将文档索引（分块 + 向量化 + 存入 ChromaDB）
    3. 执行多次检索，验证结果的相关性
    4. 展示检索结果的溯源信息
"""

import sys
sys.path.insert(0, ".")

from src.rag.parser import DocumentParser
from src.rag.retriever import Retriever


def main():
    print("=" * 60)
    print("RAG 全链路验证：Parser → Chunker → Embedder → ChromaDB")
    print("=" * 60)

    # ============================================================
    # 第一步：解析文档
    # ============================================================
    print("\n[步骤 1] 解析文档...")
    parser = DocumentParser()
    document = parser.parse("data/向量数据库入门指南.md")
    print(f"  文件：{document.metadata.get('file_name')}")
    print(f"  大小：{len(document)} 字符")

    # ============================================================
    # 第二步：索引文档（分块 + 向量化 + 存入 ChromaDB）
    # ============================================================
    print("\n[步骤 2] 索引文档...")
    retriever = Retriever(
        chunk_size=300,
        chunk_overlap=50,
    )

    # 首次索引
    indexed_count = retriever.index(document)
    print(f"  索引完成：{indexed_count} 个 Chunk 已入库。")

    # 验证去重：再次索引同一文档应被跳过
    print("\n  验证去重机制：")
    duplicate_count = retriever.index(document)
    print(f"  第二次索引被跳过（返回 {duplicate_count}）。")

    # ============================================================
    # 第三步：查看向量库状态
    # ============================================================
    print("\n[步骤 3] 向量库状态...")
    status = retriever.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # ============================================================
    # 第四步：执行检索
    # ============================================================
    print("\n[步骤 4] 执行检索测试...")

    test_queries = [
        "什么是向量数据库？",
        "余弦相似度怎么计算？",
        "分块策略有哪些？",
        "今天天气怎么样？",  # 故意问一个知识库中没有的问题
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"  查询：{query}")

        results = retriever.search(query, top_k=2)

        if len(results) == 0:
            print(f"  ⚠️  未找到相关结果（可能知识库中没有相关内容）。")
            continue

        for i, result in enumerate(results):
            print(f"  #{i+1} | 相似度={result.score:.4f}")
            print(f"      来源：{result.metadata.get('file_name', '?')}")
            print(f"      Chunk 序号：{result.metadata.get('chunk_index', '?')}")
            # 显示匹配文本的前 120 字符
            preview = result.chunk_content[:120].replace("\n", " ")
            print(f"      内容：{preview}...")

    # ============================================================
    # 第五步：验证阈值过滤机制
    # ============================================================
    print(f"\n{'=' * 60}")
    print("[步骤 5] 阈值过滤验证...")
    print(f"  查询：'今天天气怎么样？'（阈值=0.5）")

    # 用一个较高的阈值过滤掉低相关度的结果
    filtered_results = retriever.search(
        "今天天气怎么样？",
        top_k=3,
        score_threshold=0.5,  # 相似度低于 0.5 的结果会被丢弃
    )

    if len(filtered_results) == 0:
        print(f"  ✅ 正确：所有结果的相似度都低于阈值，返回空列表。")
        print(f"     这证明了「拒绝回答」机制的可行性——")
        print(f"     当知识库中确实没有相关内容时，系统能识别出来。")
    else:
        print(f"  ⚠️  返回了 {len(filtered_results)} 个结果：")
        for r in filtered_results:
            print(f"     相似度={r.score:.4f}: {r.chunk_content[:80]}...")

    print()
    print("=" * 60)
    print("✅ RAG 全链路验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
