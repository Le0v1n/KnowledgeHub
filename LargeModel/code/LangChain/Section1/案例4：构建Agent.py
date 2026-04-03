import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import chat_agent_executor

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

def get_agent_response(query):
    response = agent_executor.invoke(
        input={
            'messages': HumanMessage(
                content=query
            )
        }
    )
    # 尝试从工具调用结果里提取答案
    for message in response['messages']:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if 'answer' in tool_call.args:
                    return tool_call.args['answer']
    return response['messages'][-1].content

if __name__ == '__main__':
    # 构建LLM模型
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 构建Tavily搜索引擎工具
    search_tool: Runnable = TavilySearchResults(
        max_results=2,  # 返回结果的最大数量
        include_answer=True  # 让工具尝试直接返回答案
    )

    # 先把工具放到一起
    tools: list = [search_tool]

    # 创建一个Agent
    agent_executor: Runnable = chat_agent_executor.create_react_agent(
        model=model,
        tools=tools
    )

    # 示例1：调用Agent
    resp_1 = agent_executor.invoke(
        input={
            'messages': HumanMessage(
                content='你是谁？'
            )
        },
    )
    print(resp_1)  # 在调用的时候用的什么key，在取的时候我们就可以用这个key来取

    # 示例2：调用Agent
    resp_2 = agent_executor.invoke(
        input={
            'messages': HumanMessage(
                content='西安今天的天气怎么样？'
            )
        },
    )
    print(resp_2)  # 在调用的时候用的什么key，在取的时候我们就可以用这个key来取