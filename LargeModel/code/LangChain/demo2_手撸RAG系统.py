import requests
import PyPDF2
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===================== 配置项 =====================
API_KEY = os.getenv("SILICONFLOW_API_KEY")
API_BASE = os.getenv("API_BASE")
# 对话模型
# CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# CHAT_MODEL = "deepseek-ai/DeepSeek-R1"
# CHAT_MODEL = "Qwen/Qwen3.5-27B"
CHAT_MODEL = "Pro/MiniMaxAI/MiniMax-M2.5"
# 向量模型
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
# 向量维度（bge-large-zh是1024维，必须固定！）
VECTOR_DIM = 1024

# ======================================================================
# ===================== 环节1：手撸文档加载 =====================
def load_document(file_path: str):
    """
    纯手撸文档加载：仅支持 TXT / PDF 两种格式
    痛点：
    1. 仅支持2种格式，不支持word/md/excel
    2. 无编码自动适配、无异常处理
    3. 代码冗余，新增格式需要重写逻辑
    """
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    else:
        raise ValueError("手撸加载器仅支持 txt / pdf 格式！")
    return text

def load_documents_from_folder(folder_path: str):
    """
    从文件夹加载所有支持的文档
    支持格式：.txt, .pdf
    返回：(merged_text, file_count) 合并的文本内容和加载的文件数量
    """
    import os
    
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
        raise ValueError(f"文件夹中未找到支持的文档格式 ({', '.join(supported_extensions)})")
    
    merged_text = "\n\n".join(all_texts)
    return merged_text, len(loaded_files)

# ===================== 环节2：手撸文本切分（痛点：切分愚蠢） =====================
def split_text(text: str, chunk_size=500, chunk_overlap=50):
    """
    纯手撸文本切分：按固定字符长度切分，无语义感知
    痛点：
    1. 强行切断句子/段落，破坏上下文语义
    2. 仅支持固定长度，无法自适应文本
    3. 无标点/换行优化，切分结果混乱
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 重叠切分
        start = end - chunk_overlap
    return chunks

# ===================== 环节3：手撸向量生成 + Faiss存储/检索（痛点：手动适配繁琐） =====================
def get_embedding(text: str):
    """调用硅基API生成向量（手撸调用逻辑）"""
    url = f"{API_BASE}/embeddings"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"model": EMBED_MODEL, "input": text}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["data"][0]["embedding"]

def build_faiss_index(chunks: list):
    """
    纯手撸Faiss索引构建：手动处理向量、手动创建索引
    痛点：
    1. 必须手动指定向量维度，错一个数字就报错
    2. 无向量持久化、无批量处理
    3. 检索逻辑需要手动写，无封装
    """
    # 批量生成向量
    vectors = []
    for chunk in chunks:
        vec = get_embedding(chunk)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.float32)

    # 手动构建Faiss索引
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)
    return index, chunks

def search_faiss(index, chunks, query, top_k=3):
    """手撸检索逻辑：手动查询、手动匹配文本块"""
    query_vec = get_embedding(query)
    query_vec = np.array([query_vec], dtype=np.float32)
    # 手动执行检索
    distances, indices = index.search(query_vec, top_k)
    # 手动匹配结果
    result_chunks = [chunks[i] for i in indices[0]]
    return result_chunks

# ===================== 环节4：手撸问答逻辑=====================
def chat_with_rag(query: str, index, chunks):
    """
    纯手撸RAG问答：手动拼接Prompt、手动调用API
    痛点：
    1. 无异常处理（API超时/报错直接崩溃）
    2. 模型切换需要硬改代码，无灵活性
    3. 无历史记录、无参数配置
    """
    # 1. 检索相关文档
    context_chunks = search_faiss(index, chunks, query)
    context = "\n".join(context_chunks)

    # 2. 手动拼接Prompt（核心！RAG的关键步骤）
    prompt = f"""
    你是一个文档问答助手，请根据以下参考文档回答用户问题，不要编造答案。
    参考文档：{context}
    用户问题：{query}
    回答：
    """

    # 3. 手动调用硅基聊天API
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=data)
    answer = response.json()["choices"][0]["message"]["content"]
    return answer, context_chunks

# ======================================================================
# ===================== 运行手撸RAG系统 =====================
if __name__ == "__main__":
    # 文档文件夹路径
    DOCS_FOLDER = "docs"
    
    # 1. 从文件夹加载所有文档
    print(f"📂 正在从文件夹加载文档: {DOCS_FOLDER}")
    doc_text, file_count = load_documents_from_folder(DOCS_FOLDER)
    print(f"✅ 文档加载完成，共 {file_count} 个文件")
    
    # 2. 切分文本
    text_chunks = split_text(doc_text)
    print(f"✅ 文本切分完成，共 {len(text_chunks)} 个文本块")

    # 3. 构建Faiss向量库
    faiss_index, chunks = build_faiss_index(text_chunks)
    print("✅ Faiss向量库构建完成")

    # 4. 开始RAG问答
    questions = [
        '介绍一下Lyle',
        '你是谁？'
    ]
    
    for question in questions:
        answer, context = chat_with_rag(question, faiss_index, chunks)
        print("\n📝 回答：", answer)
        # print("\n🔍 参考文档：", context)
    