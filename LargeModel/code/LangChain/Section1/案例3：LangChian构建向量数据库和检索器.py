import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 准备测试数据，假设我们提供的文档数据如下：
my_documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",  # 文档的核心文本内容
        metadata={"source": "哺乳动物宠物文档"},  # 元数据
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

if __name__ == '__main__':
    # 实例化向量数据库（向量空间）
    vector_store = Chroma.from_documents(
        documents=my_documents,
        embedding=OpenAIEmbeddings(base_url="https://www.dmxapi.com/v1")  # 使用什么应该的Embedding方式
    )

    # 根据向量空间创建一个检索器
    retriever = RunnableLambda(
        func=vector_store.similarity_search
    ).bind(k=1)  # k=1表示选取相似度最高的那个

    # 定义Prompt Template
    message = """使用提供的上下文（向量空间/向量数据库）来回答这个问题：{question}。
    上下文：{context}"""

    prompt_template = ChatPromptTemplate.from_messages(
        messages=[  # 需要传入Sequence，即tuple、list、str
            ('human', message)
        ]  
    )

    # 构建LLM模型
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 构建chain（因为我们在prompt_template定义了message，所以我们应该在构建chain的时候应该是先将question和context传入）
    chain = {
        # ⚠️这里的RunnablePassThrough应该加上()
        'question': RunnablePassthrough(),  # RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model
        'context': retriever,
    } | prompt_template | model

    resp = chain.invoke(
        input='请介绍一下猫'
    )
    print(f"{resp.content = }")