import os
import bs4
import langchain
import langchain.chains
import langchain.chains.sql_database
import langchain.chains.sql_database.query
import langchain_community
import langchain_community.utilities
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
from urllib.parse import quote_plus

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

if __name__ == '__main__':
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # SQLAlchemy初始化MySQL数据库连接
    HOSTNAME = '127.0.0.1'
    PORT = '3306'
    DATABASE = 'Test_Le0v1n'
    USERNAME = 'root'
    PASSWORD = 'MySQL@1024'

    # 对用户名和密码进行编码
    encoded_username = quote_plus(USERNAME)
    encoded_password = quote_plus(PASSWORD)

    # mysqlclient驱动URI
    MYSQL_URI = f'mysql+mysqldb://{encoded_username}:{encoded_password}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4'

    db = langchain_community.utilities.SQLDatabase.from_uri(
        database_uri=MYSQL_URI
    )

    # # 测试连接是否成功
    # print(db.get_usable_table_names())
    # print(db.run('select * from people limit 20;'))

    # 直接使用大模型和数据整合
    test_chain = langchain.chains.sql_database.query.create_sql_query_chain(
        llm=model,
        db=db
    )
    response = test_chain.invoke(
        input={
            'question': '请问people表中有多少个人？'
        }
    )
    print(response)

