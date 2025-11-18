from langserve import RemoteRunnable

if __name__ == '__main__':
    # 创建客户端
    cilent = RemoteRunnable(
        url="http://127.0.0.1:8080/chainDemo/",
    )

    # 调用服务
    result = cilent.invoke(
        input={
            "language": "English",
            "text": "很高兴见到你！"
        }
    )

    print(f"{result = }")