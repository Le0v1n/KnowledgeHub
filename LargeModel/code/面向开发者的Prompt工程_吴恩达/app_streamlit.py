import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. 初始化客户端
client = OpenAI(
    api_key=os.getenv("APIKEY"),  # 建议放入环境变量或 st.secrets
    base_url=os.getenv("BASEURL")
)

# 2. 系统预设 Prompt
SYSTEM_PROMPT = """
你是订餐机器人，为披萨餐厅自动收集订单信息。
你要首先问候顾客。然后等待用户回复收集订单信息。收集完信息需确认顾客是否还需要添加其他内容。
最后需要询问是否自取或外送，如果是外送，你要询问地址。
最后告诉顾客订单总金额，并送上祝福。

请确保明确所有选项、附加项和尺寸，以便从菜单中识别出该项唯一的内容。
你的回应应该以简短、非常随意和友好的风格呈现。

菜单包括：

菜品：
意式辣香肠披萨（大、中、小） 12.95、10.00、7.00
芝士披萨（大、中、小） 10.95、9.25、6.50
茄子披萨（大、中、小） 11.95、9.75、6.75
薯条（大、小） 4.50、3.50
希腊沙拉 7.25

配料：
奶酪 2.00
蘑菇 1.50
香肠 3.00
加拿大熏肉 3.50
AI酱 1.50
辣椒 1.00

饮料：
可乐（大、中、小） 3.00、2.00、1.00
雪碧（大、中、小） 3.00、2.00、1.00
瓶装水 5.00
"""

# 3. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# 4. 界面标题
st.title("🍕 披萨订餐机器人")

# 5. 渲染历史消息（跳过 System）
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. 处理用户输入
if prompt := st.chat_input("请输入您的需求..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        def stream_generator():
            try:
                with client.chat.completions.stream(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages,
                    temperature=0
                ) as stream:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    final = stream.get_final_response().choices[0].message.content
                    yield ""
                    return final
            except Exception:
                try:
                    collected = ""
                    for chunk in client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.messages,
                        temperature=0,
                        stream=True
                    ):
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            text = chunk.choices[0].delta.content
                            collected += text
                            yield text
                    return collected
                except Exception:
                    resp = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.messages,
                        temperature=0
                    )
                    return resp.choices[0].message.content

        with st.chat_message("assistant"):
            full_text = st.write_stream(stream_generator())
        st.session_state.messages.append({"role": "assistant", "content": full_text})
    except Exception as e:
        st.error(f"调用出错: {e}")
