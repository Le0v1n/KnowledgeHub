#!/usr/bin/env python
"""
demo3_使用LangChain实现RAG.py - 修复版
核心修复：使用与demo2一致的固定长度切分，而非LangChain语义切分
"""
import os
import warnings
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv

# 只用LangChain的LLM和Embeddings，不使用其文本切分和FAISS封装
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

warnings.filterwarnings("ignore")

# ===================== 加载环境变量 & 配置 =====================
load_dotenv()
API_KEY = os.getenv("SILICONFLOW_API_KEY")
API_BASE = os.getenv("API_BASE")

CHAT_MODEL = "Pro/MiniMaxAI/MiniMax-M2.5"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

# ===================== 环节1：文档加载（与demo2一致） =====================
def load_document(file_path: str):
    """使用PyPDF2直接读取，与demo2完全一致"""
    import PyPDF2
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    return text

def load_documents_from_folder(folder_path: str):
    """从文件夹加载所有文档并合并为纯文本（与demo2一致）"""
    supported_extensions = ('.txt', '.pdf')
    all_texts = []
    loaded_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    text = load_document(file_path)
                    all_texts.append(text)
                    loaded_files.append(file)
                    print(f"  ✅ 加载: {file}")
                except Exception as e:
                    print(f"  ❌ 加载失败: {file} - {str(e)}")
    
    if not loaded_files:
        raise ValueError(f"文件夹中未找到支持的文档格式")
    
    merged_text = "\n\n".join(all_texts)
    return merged_text, len(loaded_files)

# ===================== 环节2：文本切分（关键修复：使用demo2的固定长度切分） =====================
def split_text(text: str, chunk_size=500, chunk_overlap=50):
    """
    使用demo2的固定长度切分方式，与demo2完全一致
    这是关键修复：不使用LangChain的语义切分，避免改变chunk结构
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

# ===================== 环节3：向量存储（修复版 - 使用纯FAISS，避免LangChain FAISS的映射问题） =====================
class SimpleVectorStore:
    """
    简化的向量存储类，直接使用FAISS，避免LangChain FAISS分批添加时的索引映射问题
    """
    def __init__(self, index, texts, embeddings):
        self.index = index
        self.texts = texts
        self.embeddings = embeddings
    
    def similarity_search(self, query, k=3):
        """执行相似性搜索"""
        query_vec = self.embeddings.embed_query(query)
        query_vec = np.array([query_vec], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "content": self.texts[idx],
                "distance": float(distances[0][i])
            })
        return results
    
    def save_local(self, save_path):
        """保存到本地"""
        import os
        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, f"{save_path}/index.faiss")
        with open(f"{save_path}/texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        print(f"✅ 向量库保存完成：{save_path}")
    
    @classmethod
    def load_local(cls, save_path, embeddings):
        """从本地加载"""
        index = faiss.read_index(f"{save_path}/index.faiss")
        with open(f"{save_path}/texts.pkl", "rb") as f:
            texts = pickle.load(f)
        return cls(index, texts, embeddings)


def create_faiss_vectorstore(chunks, save_path="faiss_siliconflow_index", batch_size=32):
    """
    创建FAISS向量存储（修复版）
    使用SimpleVectorStore避免LangChain FAISS分批添加时的索引映射问题
    """
    embeddings = OpenAIEmbeddings(
        api_key=API_KEY,
        base_url=API_BASE,
        model=EMBED_MODEL
    )

    print(f"📦 开始生成向量库（共 {len(chunks)} 个文档，分批处理）...")
    
    # 分批生成向量
    all_vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"  处理第 {i+1}-{min(i+batch_size, len(chunks))} 个文档...")
        batch_vectors = embeddings.embed_documents(batch)
        all_vectors.extend(batch_vectors)
    
    # 构建FAISS索引
    vectors_array = np.array(all_vectors, dtype=np.float32)
    dimension = vectors_array.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_array)
    
    print(f"✅ 索引构建完成：{index.ntotal} 个向量，维度 {dimension}")
    
    # 创建SimpleVectorStore
    vectorstore = SimpleVectorStore(index, chunks, embeddings)
    vectorstore.save_local(save_path)
    
    return vectorstore


# ===================== 环节4：问答 =====================
def build_llm():
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        model=CHAT_MODEL,
        temperature=0.1,
        max_tokens=1024
    )


def chat_with_rag_stream(llm, vectorstore, question: str):
    print(f"\n❓ 问题：{question}")
    
    # 使用SimpleVectorStore进行检索
    results = vectorstore.similarity_search(question, k=3)
    print(f"📖 检索到 {len(results)} 个相关文档")
    
    if results:
        for idx, result in enumerate(results):
            has_lyle = "Lyle" in result['content'] or "lyle" in result['content'].lower()
            print(f"  {idx+1}. {'✅' if has_lyle else '❌'} (dist: {result['distance']:.4f}): {result['content'][:100]}...")
        context = "\n".join([r["content"] for r in results])
    else:
        context = "没有找到相关文档。"
    
    prompt = f"""请根据以下参考文档回答用户问题，不要编造答案。

参考文档：
{context}

用户问题：{question}
回答："""
    
    print("\n💡 回答：", end="", flush=True)
    
    full_answer = ""
    for chunk in llm.stream(prompt):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_answer += chunk.content
    
    print("\n")
    return full_answer


# ===================== 主函数 =====================
if __name__ == "__main__":
    DOCS_FOLDER = "docs"
    
    print(f"📂 正在从文件夹加载文档: {DOCS_FOLDER}")
    doc_text, file_count = load_documents_from_folder(DOCS_FOLDER)
    print(f"✅ 文档加载完成，共 {file_count} 个文件")
    
    # 关键修复：使用demo2的固定长度切分，而非LangChain语义切分
    text_chunks = split_text(doc_text)
    print(f"✅ 文本切分完成，共 {len(text_chunks)} 个文本块")
    
    vectorstore = create_faiss_vectorstore(text_chunks)
    llm = build_llm()

    questions = [
        "介绍一下Lyle",
        "你是谁？"
    ]
    
    for question in questions:
        chat_with_rag_stream(llm, vectorstore, question)
