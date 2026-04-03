from sentence_transformers import SentenceTransformer, util

# 加载多语言句子向量模型（适配中文）
st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 待计算的句子列表
sentences = [
    "我爱自然语言处理",
    "我喜欢NLP",
    "今天天气很好"
]

# 生成句子语义向量
sentence_embeddings = st_model.encode(sentences)

# 计算余弦相似度
sim_1_2 = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])
sim_1_3 = util.cos_sim(sentence_embeddings[0], sentence_embeddings[2])

# 打印结果
print("\n===== 句子相似度计算结果 =====")
print(f"'{sentences[0]}' 和 '{sentences[1]}' 的相似度：{sim_1_2.item():.4f}")
print(f"'{sentences[0]}' 和 '{sentences[2]}' 的相似度：{sim_1_3.item():.4f}")