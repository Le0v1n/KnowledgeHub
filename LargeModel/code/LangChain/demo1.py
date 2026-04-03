# 导入核心依赖
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. 验证Ollama调用（使用我们部署过的模型，比如llama3）
llm = OllamaLLM(model="modelscope.cn/Qwen/Qwen3-8B-GGUF:Q4_K_M")
print(llm.invoke("你好, LangChain!"))  # 输出正常即说明Ollama对接成功

# 2. 验证Faiss对接（创建简单向量库）
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # 轻量级Embedding模型
texts = ["LangChain是一个大模型应用框架", "RAG是检索增强生成技术"]
db = FAISS.from_texts(texts, embeddings)  # 创建Faiss向量库
print(f"Faiss向量库创建成功，向量数量：{db.index.ntotal}")  # 输出2即说明Faiss对接成功