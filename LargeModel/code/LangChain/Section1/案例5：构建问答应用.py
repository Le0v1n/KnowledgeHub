import os
import bs4
import langchain
import langchain.chains
import langchain_community
import langchain_core
import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import langchain_community.chat_message_histories
import langchain_core.runnables
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

def get_session_history(session_id: str):
    """
    此函数的作用是根据给定的 session_id 获取对应的会话历史记录。
    如果指定的 session_id 不存在于 vector_store 中，会为该 session_id 创建一个新的会话历史记录。

    Args:
        session_id (str): 用于标识特定会话的唯一字符串。

    Returns:
        ChatMessageHistory: 对应 session_id 的会话历史记录对象。
    """
    # 检查传入的 session_id 是否不在 vector_store 这个数据结构中
    if session_id not in vector_store:
        # 如果 session_id 不在 vector_store 中，为该 session_id 创建一个新的 ChatMessageHistory 对象
        # 并将其存储在 vector_store 中，键为 session_id
        vector_store[session_id] = langchain_community.chat_message_histories.ChatMessageHistory()
    # 返回 vector_store 中对应 session_id 的会话历史记录对象
    return vector_store[session_id]

if __name__ == '__main__':
    # 创建文档加载器的类
    content_loader = WebBaseLoader(
        web_path=[
            'https://lilianweng.github.io/posts/2023-06-23-agent/',
        ],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=(  # 指定哪些元素是需要读取的
                    'post-header',
                    'post-title',
                    'post-content'
                )
            )
        )
    )

    # 正式加载文档 -> 一篇博客内容的博客
    docs: list = content_loader.load()

    # 大文本的切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个文本块的最大字符数量。文本会按这个大小进行分割，从而生成多个文本块
        chunk_overlap=200  # 相邻文本块之间重叠的字符数量。其作用是保证文本块之间有一定的连续性，防止关键信息在分割时被截断
    )
    # 正式切割
    splits: list = splitter.split_documents(documents=docs)

    # 存储：将文本转换为向量
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(base_url="https://www.dmxapi.com/v1"),
    )

    # 创建检索器
    retriever = vector_store.as_retriever()

    # 创建问题的Prompt模板
    system_prompt = """你是负责问答任务的助手。请使用检索到的上下文信息回答问题。
    若无法确定答案，请直接说明未知。
    答案需简洁，最多三句话。\n
    
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ('system', system_prompt),  # LLM初始化时看到的Prompt
            MessagesPlaceholder('chat_history'),  # 历史聊天记录（这里用占位符先占住）
            ('human', "{input}")  # 用户输入的Prompt
        ]
    )

    # 构建LLM模型
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 创建文档组合链
    document_combine_chain: Runnable = langchain.chains.combine_documents.create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # 创建子链
    ## 子链的Prompt Template
    contextualize_q_system_prompt: str = """给定一段聊天记录和用户的最新问题（该问题可能涉及聊天记录中的上下文），
    请构造一个无需依赖聊天记录即可理解的独立问题。
    请勿回答该问题，仅在需要时对其进行重新表述，否则直接按原样返回。"""

    retriever_history_template = ChatPromptTemplate.from_messages(
        messages=[
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    ## 创建历史记录链
    history_chain = langchain.chains.history_aware_retriever.create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_history_template
    )

    # 保存聊天的历史记录
    vector_store: dict = {}

    # 创建终极链: 将前面创建的两个链整合在一起
    chain = langchain.chains.retrieval.create_retrieval_chain(
        retriever=history_chain,
        combine_docs_chain=document_combine_chain
    )

    result_chain = langchain_core.runnables.RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key='input',  # 用户输入的key
        history_messages_key='chat_history',  # 历史记录的key
        output_messages_key='answer'  # 返回结果的key
    )

    # 第一轮对话
    resp_1 = result_chain.invoke(
        input={'input': '什么是任务分解？'},
        config={'configurable': {'session_id': 'Le0v1n_001'}}
    )
    print(f"---------- 第一轮对话 ----------\n{resp_1['answer']}\n")

    # 第二轮对话 -> 考验是否有上下文能力
    resp_2 = result_chain.invoke(
        input={'input': '它有哪些通用的方法呢？'},
        config={'configurable': {'session_id': 'Le0v1n_001'}}  # 这里的session_id不能变
    )
    print(f"---------- 第二轮对话 ----------\n{resp_2['answer']}\n")

    # 额外对话（修改了会话id -> 没有上下文能力）
    resp_3 = result_chain.invoke(
        input={'input': '它有哪些通用的方法呢？'},
        config={'configurable': {'session_id': 'Le0v1n_002'}}  # 这里的session_id不能变
    )
    print(f"---------- 额外对话 ----------\n{resp_3['answer']}\n")