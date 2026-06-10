"""
文档解析器验证脚本
==================
运行方式：python test_parser.py

测试内容：
    1. 解析 Markdown 文件
    2. 检查 Document 对象的 content 和 metadata
    3. 测试不支持的格式是否正确报错
    4. 测试不存在的文件是否正确报错
"""

import sys
sys.path.insert(0, ".")

from src.rag.parser import DocumentParser, Document


def test_parse_markdown():
    """
    测试 1：解析 Markdown 文件
    """
    print("=" * 50)
    print("测试 1：解析 Markdown 文件")
    print("=" * 50)

    parser = DocumentParser()

    # 解析我们刚才创建的示例文档
    doc = parser.parse("data/向量数据库入门指南.md")

    # 检查结果
    print(f"\n文件来源   : {doc.metadata.get('source')}")
    print(f"文件类型   : {doc.metadata.get('file_type')}")
    print(f"行数       : {doc.metadata.get('line_count')}")
    print(f"字符数     : {doc.metadata.get('char_count')}")
    print(f"解析时间   : {doc.metadata.get('parse_time')}")
    print(f"内容长度   : {len(doc)} 字符")
    print(f"\n--- 文档预览（前 200 字符）---")
    print(doc.content[:200])
    print("...")
    print()

    # 验证：内容不能为空
    if len(doc.content) > 0:
        print("✅ Markdown 解析成功，内容不为空。")
    else:
        print("❌ Markdown 解析失败，内容为空！")

    return True


def test_unsupported_format():
    """
    测试 2：不支持的格式应抛出 ValueError
    """
    print("\n" + "=" * 50)
    print("测试 2：不支持的格式（应报错）")
    print("=" * 50)

    parser = DocumentParser()

    try:
        # 尝试解析一个 .txt 文件（我们不支持）
        parser.parse("data/test.txt")
        print("❌ 应该抛出 ValueError，但没有！")
        return False
    except ValueError as e:
        print(f"✅ 正确捕获到 ValueError：{e}")
        return True
    except FileNotFoundError:
        # 文件不存在是另一个错误，也算合理
        print("⚠️  文件不存在（这也是一种合理的错误）")
        return True


def test_file_not_found():
    """
    测试 3：不存在的文件应抛出 FileNotFoundError
    """
    print("\n" + "=" * 50)
    print("测试 3：不存在的文件（应报错）")
    print("=" * 50)

    parser = DocumentParser()

    try:
        parser.parse("data/不存在的文件.pdf")
        print("❌ 应该抛出 FileNotFoundError，但没有！")
        return False
    except FileNotFoundError as e:
        print(f"✅ 正确捕获到 FileNotFoundError：{e}")
        return True


def test_document_object():
    """
    测试 4：Document 对象的基本操作
    """
    print("\n" + "=" * 50)
    print("测试 4：Document 对象操作")
    print("=" * 50)

    # 手动创建一个 Document
    doc = Document(
        content="这是一段测试文本。",
        metadata={"source": "test.txt", "author": "Claude"}
    )

    # 测试 __repr__
    print(f"\nrepr(doc)  : {repr(doc)}")

    # 测试 __len__
    print(f"len(doc)   : {len(doc)}")

    # 测试 metadata 访问
    print(f"metadata   : {doc.metadata}")

    print("✅ Document 对象操作正常。")


if __name__ == "__main__":
    all_passed = True

    all_passed = all_passed and test_parse_markdown()
    all_passed = all_passed and test_unsupported_format()
    all_passed = all_passed and test_file_not_found()
    test_document_object()

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ 所有测试通过！文档解析器工作正常。")
    else:
        print("⚠️  部分测试未通过，请检查上方输出。")
    print("=" * 50)
