import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# 1. 初始化客户端
client = OpenAI(
    api_key=os.getenv("APIKEY"),
    base_url=os.getenv("BASEURL")
)

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    response = get_completion("Take the letters in lollipop and reverse them")
    print(response)