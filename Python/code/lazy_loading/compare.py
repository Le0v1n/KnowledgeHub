import os
import time
from PIL import Image

def method_1():
    # 使用列表推导式直接读取所有图片并保存到列表中
    t_begin = time.perf_counter()
    images = [Image.open(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    t_loading = time.perf_counter()
    print(f"[方法1] [加载耗时] {t_loading - t_begin:.6f}s")
    print(f"[方法1] [图片数量] {len(images)}")
    print(f"[方法1] [第一张图片] {images[0]}")

    # 遍历图片列表并逐一处理
    for img in images:
        # 将图片转换为灰度图
        img_gray = img.convert("L")
    t_end= time.perf_counter()
    print(f"[方法1] [程序总耗时] {t_end - t_begin:.6f}s")

def method_2():
    # 使用列表推导式直接读取所有图片并保存到列表中
    t_begin = time.perf_counter()
    images_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    t_loading = time.perf_counter()
    print(f"[方法2] [加载耗时] {t_loading - t_begin:.6f}s")
    print(f"[方法2] [图片数量] {len(images_paths)}")
    print(f"[方法2] [第一张图片] {images_paths[0]}")

    # 遍历图片列表并逐一处理
    for path in images_paths:
        with Image.open(path) as img:  # 使用 with 确保文件正确关闭
            # 将图片转换为灰度图
            img_gray = img.convert("L")
    t_end= time.perf_counter()
    print(f"[方法2] [程序总耗时] {t_end - t_begin:.6f}s")

def method_3_lazy_load():
    # 定义如何加载图片的函数
    def lazy_load_images(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                yield os.path.join(folder_path, file)

    # 遍历图片列表并逐一处理
    t_begin = time.perf_counter()
    for path in lazy_load_images(folder_path):
        t_loading = time.perf_counter()
        with Image.open(path) as img:
            img_gray = img.convert("L")

    t_end= time.perf_counter()
    print(f"[方法3-Lazy Loading] [程序总耗时] {t_end - t_begin:.6f}s")


if __name__ == '__main__':
    # 定义图片文件夹路径
    folder_path = "test-Le0v1n/images_10k"

    method_1()
    print()

    method_2()
    print()

    method_3_lazy_load()