from langchain_community.document_loaders import TextLoader, PyPDFLoader

# ====================== 场景1：加载纯文本 TXT 文件 ======================
# 初始化加载器（传入文件路径）
txt_loader = TextLoader(
    file_path="docs/RAG系统测试QA（纯问答，用于系统加载检验）.txt", 
    encoding="utf-8"  # encoding解决中文乱码
)

# 加载文档 → 输出：List[Document]（列表，支持多文件/多页码）
txt_docs = txt_loader.load()
print(f"{type(txt_docs) = }")
print(f"{len(txt_docs) = }")

# 打印结果
print("===== TXT 加载结果 =====")
print("文本内容：", txt_docs[0].page_content[:100])  # 取前100字符
print("元数据：", txt_docs[0].metadata)  # 自动生成：文件路径

# ====================== 场景2：加载 PDF 文件 ======================
# 初始化加载器
pdf_loader = PyPDFLoader(
    file_path="docs/RAG系统测试QA（纯问答，用于系统加载检验）.pdf"
)

# 加载文档 → 自动拆分页码，每个页码是一个 Document
pdf_docs = pdf_loader.load()

# 打印结果
print("\n===== PDF 加载结果 =====")
print("总页数：", len(pdf_docs))
print("第1页文本：", pdf_docs[0].page_content[:100])
print("第1页元数据：", pdf_docs[0].metadata)  # 自动生成：文件路径+页码