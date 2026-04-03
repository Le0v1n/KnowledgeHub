import faiss


# 三个核心参数
dim     = 64               # 向量维度
param   = 'PQ16'           # 把向量切成16段（64/16=4，每段4维）
measure = faiss.METRIC_L2  # 距离度量方式（常用L2或内积）

# 构建索引
index = faiss.index_factory(dim, param, measure)