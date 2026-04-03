import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载开源语义模型（把文本转成向量，不用自己造随机向量了）
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 2. 准备测试数据（模拟你的本地笔记/文档）
docs = []
docs.append("Faiss是Meta开源的向量检索库，支持多种索引类型，用于高维向量的快速相似性检索",)
docs.append("IVF索引通过k-means分桶实现检索加速，需要先训练聚类中心",)
docs.append("PQ乘积量化通过向量分段压缩，大幅减少内存占用，适合超大规模向量库",)
docs.append("HNSW是基于分层图的索引，检索速度极快，召回率接近暴力检索，适合线上实时场景",)
docs.append("Flat索引是暴力检索，召回率100%，适合小数据量场景",)
docs.append("Python是一门解释型编程语言，广泛用于机器学习、数据分析领域",)
docs.append("向量数据库的核心是向量检索引擎，很多底层基于Faiss实现",)
docs.append("RAG检索增强生成的核心步骤是：文档向量化、向量检索、prompt拼接、大模型生成",)
docs.append("Faiss的核心的索引有Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW",)  # 正确答案1
docs.append("Faiss的核心的索引有6个，分别是Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW",)  # 正确答案2
docs.append("Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW是Faiss的核心的索引",)  # 正确答案3

def read_txt_file(file_path):
    """读取txt文件内容并返回文本列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 按行读取并去除空行和首尾空白字符
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []


docs.extend(list(set(read_txt_file("小说8万字大纲.txt"))))

print(f"{docs[0:10] = }")

# 3. 把文本转成向量
doc_vectors = model.encode(docs).astype("float32")

# 4. 使用HNSW构建索引
dim = doc_vectors.shape[1]  # 新增：自动获取向量的真实维度
index = faiss.index_factory(dim, "HNSW32", faiss.METRIC_L2)

# 因为HNSW的特性，不需要训练，直接添加向量即可
index.add(doc_vectors)
print(f"索引构建完成，向量总数：{index.ntotal}")


# 5. 执行索引（需要真正用Faiss了）
query = "Faiss的索引类型有哪些？"

# 因为是向量检索，所以需要把查询文本也转成向量
query_vector = model.encode([query]).astype("float32")

# 检索Top3最相似的文本
k = 100
D, I = index.search(query_vector, k)

# 6. 打印查询结果
print(f"我们的查询是: {query}")
print(f"在索引中检索到的最相关的内容为（top {k}）:")
for idx, (distance, doc_id) in enumerate(zip(D[0], I[0])):
    print(f"排名 {idx+1}: 距离={distance:.4f}, 文档ID={doc_id}, 内容={docs[doc_id]}")