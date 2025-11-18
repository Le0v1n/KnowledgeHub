from langchain_core.documents import Document

# 创建一个 Document 对象
doc = Document(
    page_content="这是一段示例文本，用于演示 Document 类的使用。",
    metadata={"source": "示例文档", "date": "2024-01-01"}
)

# 打印文档内容和元数据
print("文档内容:", doc.page_content)
print("文档元数据:", doc.metadata)
