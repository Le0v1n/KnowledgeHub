from unittest import result


def read_txt_file(file_path):
    """读取txt文件内容并返回文本列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 按行读取并去除空行和首尾空白字符
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []


result = list(set(read_txt_file("小说8万字大纲.txt")))
print(result)
print(len(result))
print(type(result))
