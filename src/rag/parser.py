"""
文档解析器模块 (Document Parser)
===================================

核心职责：
    将 PDF 和 Markdown 文件解析为统一的「文档对象 (Document)」，
    使得后续的分块、向量化、检索模块不需要关心原始文件格式。

设计原则：
    1. 统一输出格式：无论什么文件类型，最终都产出 Document 对象。
    2. 保留元数据：记录文件名、页数、解析时间等信息，用于溯源。
    3. 容错设计：解析失败时不崩溃，返回空内容并记录错误信息。

关于 Document 这个简单的数据容器：
    它是一个「Plain Old Python Object」（普通 Python 对象），
    只负责装数据，不包含任何业务逻辑。
    用简单的类而不是字典，是为了让字段有明确的定义和 IDE 自动补全。
"""

import os
from datetime import datetime


# ============================================================
# 第一步：定义统一的文档数据容器
# ============================================================

class Document:
    """
    解析后的文档对象。

    属性说明：
        content  : str  —— 文档的纯文本内容（核心数据）
        metadata : dict —— 文档的附加信息（来源、页数、解析时间等）
    """

    def __init__(self, content: str, metadata: dict = None):
        """
        创建一个文档对象。

        参数：
            content  : 从 PDF/Markdown 提取出的纯文本
            metadata : 一个字典，包含文档的附加信息。
                       如果调用时不传这个参数，默认使用空字典。
        """
        self.content = content

        # 如果调用者传了 metadata 就用它，否则用一个空字典
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def __repr__(self):
        """
        定义文档对象的「可读表示」。

        当你 print(doc) 时，Python 会调用这个方法。
        这里我们显示前面 100 个字符的预览，而不是打印全部内容。
        """
        preview = self.content[:100].replace("\n", " ")
        if len(self.content) > 100:
            preview = preview + "..."
        return f"Document(source='{self.metadata.get('source', 'unknown')}', preview='{preview}')"

    def __len__(self):
        """
        定义 len(doc) 的返回值 —— 返回文档内容的字符数。

        这样你就可以用 len(doc) 快速判断文档大小。
        """
        return len(self.content)


# ============================================================
# 第二步：定义解析器类
# ============================================================

class DocumentParser:
    """
    文档解析器。

    支持的文件格式：
        - PDF     (.pdf)  —— 使用 pypdf 库提取文本
        - Markdown (.md)   —— 保留结构地读取纯文本

    使用方法：
        parser = DocumentParser()
        doc = parser.parse("path/to/file.pdf")   # 自动识别格式
        doc = parser.parse("path/to/file.md")    # 自动识别格式
    """

    # 支持的文件扩展名映射
    SUPPORTED_EXTENSIONS = {
        ".pdf": "PDF 文档",
        ".md":  "Markdown 文档",
    }

    def parse(self, file_path: str) -> Document:
        """
        解析一个文件，自动根据扩展名选择对应的解析方法。

        这是对外的统一入口——调用者不需要关心文件类型。

        参数：
            file_path : 文件的完整路径（如 "data/paper.pdf"）

        返回值：
            一个 Document 对象，包含提取的文本和元数据

        异常处理：
            文件不存在 → 抛出 FileNotFoundError
            不支持的格式 → 抛出 ValueError
            解析过程出错 → 返回一个内容为空但带有错误信息的 Document
        """
        # --- 检查 1：文件是否存在 ---
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        # --- 检查 2：文件扩展名是否支持 ---
        # os.path.splitext 把 "data/paper.pdf" 拆成 ("data/paper", ".pdf")
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()  # 统一转为小写（.PDF → .pdf）

        if extension not in self.SUPPORTED_EXTENSIONS:
            supported_list = ", ".join(self.SUPPORTED_EXTENSIONS.keys())
            raise ValueError(
                f"不支持的文件格式 '{extension}'。"
                f"当前支持：{supported_list}"
            )

        # --- 根据扩展名分发到对应的解析方法 ---
        try:
            if extension == ".pdf":
                return self._parse_pdf(file_path)
            elif extension == ".md":
                return self._parse_markdown(file_path)
        except Exception as error:
            # 解析过程中出现任何异常，返回一个带有错误信息的空文档
            # 这样上层调用者不需要 try/except，只需检查 metadata 中的 error 字段
            return Document(
                content="",
                metadata={
                    "source": file_path,
                    "error": f"{type(error).__name__}: {str(error)}",
                    "parse_time": datetime.now().isoformat(),
                }
            )

    # ============================================================
    # PDF 解析
    # ============================================================

    def _parse_pdf(self, file_path: str) -> Document:
        """
        解析 PDF 文件，逐页提取文本。

        关于 pypdf.PdfReader：
            它读取 PDF 的「内容流」，把每页的绘制指令中的文字提取出来。
            但它不处理图片中的文字（那是 OCR 的范畴），也不保证
            提取出的文字顺序和你在屏幕上看到的一模一样。

        参数：
            file_path : PDF 文件的路径

        返回值：
            Document 对象
        """
        # 延迟导入：只在真正需要解析 PDF 时才导入 pypdf
        # 如果用户只解析 Markdown，就不用加载这个库
        from pypdf import PdfReader

        # 用 PdfReader 打开 PDF 文件
        reader = PdfReader(file_path)

        # 逐页提取文本
        all_pages_text = []  # 存放每一页的文本
        total_pages = len(reader.pages)

        for page_index in range(total_pages):
            # 获取第 page_index 页
            page = reader.pages[page_index]

            # extract_text() 从这一页的绘制指令中提取文字
            page_text = page.extract_text()

            # 有些页可能完全是图片（没有文字），此时 extract_text() 返回 ""
            if page_text:
                all_pages_text.append(page_text)

        # 将所有页的文本用换行符拼接
        full_text = "\n\n".join(all_pages_text)

        # 构建元数据
        metadata = {
            "source": file_path,                        # 文件来源路径
            "file_name": os.path.basename(file_path),   # 纯文件名（不含路径）
            "file_type": "pdf",                         # 文件类型
            "total_pages": total_pages,                 # 总页数
            "pages_with_text": len(all_pages_text),     # 实际有文字的页数
            "parse_time": datetime.now().isoformat(),   # 解析时间
        }

        return Document(content=full_text, metadata=metadata)

    # ============================================================
    # Markdown 解析
    # ============================================================

    def _parse_markdown(self, file_path: str) -> Document:
        """
        解析 Markdown 文件。

        Markdown 本质上就是纯文本，所以解析相对简单。
        我们做的事情：
            1. 读取文件全部内容
            2. 不做过度清洗（保留标题 #、列表 - 等标记，
               因为它们包含结构信息，对后续分块和检索有帮助）

        参数：
            file_path : Markdown 文件的路径

        返回值：
            Document 对象
        """
        # 以 UTF-8 编码读取文件全部内容
        # encoding="utf-8" 明确指定编码，避免在不同系统上出现乱码
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 构建元数据
        metadata = {
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": "markdown",
            "line_count": content.count("\n") + 1,      # 总行数
            "char_count": len(content),                  # 总字符数
            "parse_time": datetime.now().isoformat(),
        }

        return Document(content=content, metadata=metadata)
