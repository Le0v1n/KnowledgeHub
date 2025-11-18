import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

def get_session_history(session_id: str) -> str:
    """
    根据会话ID获取会话历史记录。

    如果给定的会话ID在存储中不存在，那么会为该会话ID创建一个新的ChatMessageHistory实例，并存储以备后用。
    这个函数确保了每个会话ID都有一个对应的会话历史记录对象，无论是新创建的还是之前就存在的。

    参数:
        session_id (str): 会话的唯一标识符。

    返回:
        str: 对应于给定会话ID的会话历史记录。
    """
    # 检查给定的会话ID是否已存在于存储中
    if session_id not in chat_history_store:
        # 如果不存在，为该会话ID创建一个新的ChatMessageHistory实例
        chat_history_store[session_id] = ChatMessageHistory()
    # 返回与会话ID关联的会话历史记录对象
    return chat_history_store[session_id]

if __name__ == '__main__':
    # 创建LLM实例对象
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 定义提示词模板
    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
            MessagesPlaceholder(variable_name='chat_history')
        ],
        template_format='f-string'  # 如何解析{}中的内容，默认是f-string
    )

# 创建链
chain = prompt_template | model

# 保存聊天的历史记录（所有用户的聊天记录都会保存）
# key: sessionId
# value: 聊天记录
chat_history_store: dict = {}

# 创建一个RunnableWithMessageHistory实例，用于处理聊天记录
runner = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,  # 不要加括号
    input_messages_key='chat_history',  # 输入消息的键: 每次聊天时发送msg的key
)

# 配置
config = {
    'configurable': {'session_id': 'chat_example_1'}  # 给当前聊天会话设置一个session_id
}

# 第一轮聊天 -> 没有聊天记录
response = runner.invoke(
    input={
        'chat_history': [
            HumanMessage(
                content='你好，我是Le0v1n',  # 发给LLM的第一个聊天记录
            )
        ],
        'language': '中文'  # 因为之前的Template中定义了要用什么语言进行回答，所以在第一次聊天的时候我们也需要定义这个Language
    },
    config=config
)

print(f"[第一轮] {response.content = }")

# 第二轮聊天 -> 没有聊天记录
response = runner.invoke(
    input={
        'chat_history': [
            HumanMessage(
                content='请问，我的名字是什么？',  # 这个问题就可以看出来LLM会不会记住我们的名字
            )
        ],
        'language': '中文'  # 因为之前的Template中定义了要用什么语言进行回答，所以在第一次聊天的时候我们也需要定义这个Language
    },
    config=config
)

print(f"[第二轮] {response.content = }")

# 第三轮对话 -> 返回的数据是流式的
print('-' * 25 + ' [第三轮] ' + '-' * 25)
for resp in runner.stream( # 因为我们要使用流式输出，所以这里不再是invoke而是stream
    input={
        'chat_history': [
            HumanMessage(content='请给我介绍一下LangChain')
        ],
        'language': '中文'
    },
    config=config
):  # 每一次response都是一个token
    print(f"{resp.content}", end='')
print(f"")