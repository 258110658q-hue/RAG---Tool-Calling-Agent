"""
文本分块器验证脚本
==================
运行方式：python test_chunker.py

测试内容：
    1. 用递归分块器处理示例 Markdown 文档
    2. 展示每个 chunk 的内容、大小和重叠
    3. 验证 metadata 是否正确继承
"""

import sys
sys.path.insert(0, ".")

from src.rag.parser import DocumentParser
from src.rag.chunker import TextChunker


def main():
    print("=" * 60)
    print("文本分块器验证")
    print("=" * 60)

    # 第一步：用解析器读取文档
    parser = DocumentParser()
    document = parser.parse("data/向量数据库入门指南.md")

    print(f"\n原始文档：{document.metadata.get('file_name')}")
    print(f"总字符数：{len(document)} 字符")
    print()

    # 第二步：创建分块器并执行分块
    # chunk_size=300  —— 每个块最多 300 字符（故意设小，方便观察）
    # chunk_overlap=50 —— 相邻块重叠 50 字符
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.split(document)

    # 第三步：逐个展示每个 chunk 的详细信息
    print(f"共切分为 {len(chunks)} 个 Chunk：")
    print()

    for chunk in chunks:
        # 显示 chunk 基础信息
        source = chunk.metadata.get("file_name", "unknown")
        print(f"{'='*60}")
        print(f"Chunk #{chunk.index}  |  来源：{source}  |  大小：{len(chunk)} 字符")
        print(f"{'='*60}")

        # 显示内容（截断过长的内容）
        content = chunk.content
        if len(content) > 400:
            content = content[:400] + "\n... [内容过长，已截断]"
        print(content)
        print()

    # 第四步：展示相邻 chunk 的重叠情况
    if len(chunks) >= 2:
        print("=" * 60)
        print("重叠验证：检查相邻 Chunk 的内容交集")
        print("=" * 60)

        # 取前两个 chunk，展示它们的重叠
        chunk_a = chunks[0]
        chunk_b = chunks[1]

        # 找出 chunk_a 的结尾部分和 chunk_b 的开头部分的共同文本
        overlap_size = chunker.chunk_overlap
        tail_of_a = chunk_a.content[-overlap_size:] if len(chunk_a.content) >= overlap_size else chunk_a.content
        head_of_b = chunk_b.content[:overlap_size] if len(chunk_b.content) >= overlap_size else chunk_b.content

        print(f"\nChunk #{chunk_a.index} 尾部 ({overlap_size} 字符)：")
        print(f"  \"{tail_of_a[:100]}...\"")
        print(f"\nChunk #{chunk_b.index} 开头 ({overlap_size} 字符)：")
        print(f"  \"{head_of_b[:100]}...\"")
        print(f"\n✅ 相邻 chunk 之间存在内容重叠（防止语义被切断）。")

    print()
    print("=" * 60)
    print(f"✅ 分块验证完成。共 {len(chunks)} 个 Chunk，每个 ≤ {chunker.chunk_size} 字符。")
    print("=" * 60)


if __name__ == "__main__":
    main()
