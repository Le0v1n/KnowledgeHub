import faiss
import time
import numpy as np
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer

# ====================== 1. 加载模型和准备数据 ======================
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 准备Faiss相关文档
docs = [
    "Faiss是Meta开源的向量检索库，支持多种索引类型，用于高维向量的快速相似性检索",
    "IVF索引通过k-means分桶实现检索加速，需要先训练聚类中心",
    "PQ乘积量化通过向量分段压缩，大幅减少内存占用，适合超大规模向量库",
    "HNSW是基于分层图的索引，检索速度极快，召回率接近暴力检索，适合线上实时场景",
    "Flat索引是暴力检索，召回率100%，适合小数据量场景",
    "Python是一门解释型编程语言，广泛用于机器学习、数据分析领域",
    "向量数据库的核心是向量检索引擎，很多底层基于Faiss实现",
    "RAG检索增强生成的核心步骤是：文档向量化、向量检索、prompt拼接、大模型生成",
    "Faiss的核心的索引有Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW",
    "Faiss的核心的索引有6个，分别是Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW",
    "Flat、IVFx Flat、PQx、IVFxPQy、LSH、HNSW是Faiss的核心的索引",
]

def read_txt_file(file_path):
    """读取txt文件内容并返回文本列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []

# 加载小说干扰数据
docs.extend(list(set(read_txt_file("小说8万字大纲.txt"))))
print(f"文档总数：{len(docs)}")

# ====================== 2. 编码向量（显式归一化） ======================
doc_vectors = model.encode(docs).astype('float32')
faiss.normalize_L2(doc_vectors)  # 显式归一化，通用最佳实践
dim = doc_vectors.shape[1]
print(f"向量维度：{dim}")

# ====================== 3. 定义查询和Ground Truth（用于计算召回率） ======================
query = "Faiss的索引类型有哪些？"
query_vector = model.encode([query]).astype('float32')
faiss.normalize_L2(query_vector)
k = 10

# 先跑Flat，获取Ground Truth（标准答案的文档ID）
index_flat = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
index_flat.add(doc_vectors)
_, I_flat = index_flat.search(query_vector, k)
ground_truth_ids = set(I_flat[0])
print(f"Ground Truth（Flat的Top10文档ID）：{ground_truth_ids}")

# ====================== 4. 循环对比不同索引 ======================
table = PrettyTable()
# 🌟 修改1：加内存占用列
table.field_names = ["索引类型", "构建耗时(秒)", "训练耗时(秒)", "添加耗时(秒)", "内存占用(MB)", "总耗时(秒)", "Top10召回率"]

# 索引类型列表（把逗号替换成下划线，方便保存文件）
index_types = [
    ("Flat", "Flat"),
    ("IVF100,Flat", "IVF100_Flat"),
    ("PQ16", "PQ16"),
    ("IVF100,PQ16", "IVF100_PQ16"),
    ("HNSW64", "HNSW64")
]

for index_factory_str, index_name in index_types:
    print(f"\n正在测试索引：{index_name}...")
    
    # 多次实验取平均（3次）
    build_times = []
    train_times = []
    add_times = []
    memory_usages = []  # 🌟 修改2：加内存列表
    total_times = []
    recall_scores = []
    
    for _ in range(3):
        # 1. 构建索引
        build_start = time.time()
        index = faiss.index_factory(dim, index_factory_str, faiss.METRIC_INNER_PRODUCT)
        build_end = time.time()
        build_time = build_end - build_start
        build_times.append(build_time)
        
        # 2. 训练索引（如果需要）
        train_time = 0
        if not index.is_trained:
            train_start = time.time()
            index.train(doc_vectors)
            train_end = time.time()
            train_time = train_end - train_start
        train_times.append(train_time)
        
        # 3. 添加向量
        add_start = time.time()
        index.add(doc_vectors)
        add_end = time.time()
        add_time = add_end - add_start
        add_times.append(add_time)
        
        # 🌟 修改3：测量索引内存占用（关键！）
        serialized = faiss.serialize_index(index)
        memory_usage = len(serialized) / (1024 * 1024)  # 字节转MB
        memory_usages.append(memory_usage)
        
        # 4. 总耗时
        total_time = build_time + train_time + add_time
        total_times.append(total_time)
        
        # 5. 设置索引的检索参数
        if "IVF" in index_factory_str:
            index.nprobe = 20  # IVF搜索20个桶
        if "HNSW" in index_factory_str:
            index.hnsw.efSearch = 64  # HNSW的候选集大小
        
        # 6. 检索并计算召回率
        _, I = index.search(query_vector, k)
        retrieved_ids = set(I[0])
        recall = len(retrieved_ids & ground_truth_ids) / len(ground_truth_ids)
        recall_scores.append(recall)
    
    # 取平均值
    avg_build = np.mean(build_times)
    avg_train = np.mean(train_times)
    avg_add = np.mean(add_times)
    avg_memory = np.mean(memory_usages)  # 🌟 修改4：取内存平均
    avg_total = np.mean(total_times)
    avg_recall = np.mean(recall_scores)
    
    # 添加到表格
    table.add_row([
        index_name,
        f"{avg_build:.4f}",
        f"{avg_train:.4f}",
        f"{avg_add:.4f}",
        f"{avg_memory:.2f}",  # 🌟 修改5：加内存到表格
        f"{avg_total:.4f}",
        f"{avg_recall:.2%}"
    ])
    
    # 7. 保存最后一次的检索结果
    with open(f"faiss_{index_name}_results.txt", "w", encoding="utf-8") as f:
        f.write(f"我们的查询是: {query}\n")
        f.write(f"在索引中检索到的最相关的内容为（top {k}）:\n")
        for idx, (distance, doc_id) in enumerate(zip(_[0], I[0])):
            f.write(f"排名 {idx+1}: 距离={distance:.4f}, 文档ID={doc_id}, 内容={docs[doc_id]}\n")

# ====================== 5. 打印结果 ======================
print("\n" + "="*80)
print("🌟 索引性能对比（3次实验取平均）")
print("="*80)
print(table)