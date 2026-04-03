import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 构建解析器
parser = StrOutputParser()

# 定义PromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', '将下面内容翻译为{language}'),
        ('user', '{text}')
    ]
)

# 创建链
chain = prompt_template | model | parser

# 直接使用chain来调用
# print(chain.invoke(
#     {
#         'language': '英语',
#         'text': '见到你很高兴!'
#     }
# ))

# ========== 将程序部署为服务 ==========
# 创建FastAPI应用
app = FastAPI(
    title='My LangChain Server',
    version='v1.0',  # 服务的版本（我们自定义的）
    description='使用LangChain的翻译服务'
)

# 添加路由
add_routes(
    app,
    chain,
    path='/chainDemo'  # 请求路径
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有 IP
        port=8080
    )
