# 1. 创建打包脚本

```bash
#!/bin/bash
# 设置 -e 使得脚本在遇到错误时停止执行
set -e

# ============================== 必要的参数 ==============================
exported_container_name="needed_export_container_name_or_id"  # 需要被导出的容器的名称或id
image_save_name="my_custom_image_name:v25.03.03"  # 镜像需要保存的名字和版本号
tar_save_path="./my_custom_image.tar"  # 镜像的tar保存路径
# ========================================================================

# 生成基于时间的随机文件夹名
temp_file="tempfile_"$(date +%Y%m%d_%H%M%S.tar)

# 在宿主机导出容器文件
echo "[INFO] Step 1/4: Export container..."
docker export "$exported_container_name" > "$temp_file"
echo "[INFO] Step 1/4: Export container completed! "
echo ""

# 将容器文件变为镜像
echo "[INFO] Step 2/4: Import image..."
docker import "$temp_file" "$image_save_name"
echo "[INFO] Step 2/4: Import image completed!"
echo ""

# 将镜像保存为tar包
echo "[INFO] Step 3/4: Save image..."
docker save -o "$tar_save_path" "$image_save_name"
echo "[INFO] Step 3/4: Save image completed!"
echo ""

# 删除多余的tar包
echo "[INFO] Step 4/4: Cleanup temporary files..."
if [ -f "$temp_file" ]; then
    rm "$temp_file"
    echo "[INFO] Step 4/4: Temporary files cleaned up!"
else
    echo "[WARNING] Temporary file does not exist, skipping deletion."
fi
echo ""

echo "[INFO] The image has been saved in $tar_save_path"

docker rmi "$image_save_name"

echo "[INFO] The image named $image_save_name has been deleted!"
```

这里需要注意三个变量需要修改：

- `exported_container_name`：需要被导出的容器的名称或id
- `image_save_name`：镜像需要保存的名字和版本号
- `tar_save_path`：生成的镜像.tar文件保存路径

# 2. 开始执行

<div align=center>
    <img src=./imgs_markdown/2025-03-03-11-36-34.png
    width=50%></br><center></center>
</div>

# 3. 验证.tar包是否可以正常加载为镜像

首先确定我们的要加载的镜像没有存在

```bash
docker images
```

之后我们开始加载镜像：

```bash
# 语法
docker load -i <刚才我们打包好的tar包>
```

<div align=center>
    <img src=./imgs_markdown/2025-03-03-11-39-59.png
    width=80%></br><center></center>
</div>

# 4. 验证加载的镜像是否可以正常创建容器

```bash
docker run -it --name <容器的名称> <镜像命令:版本/镜像id>
```

```bash
# 示例
docker run -it --name <容器的名称>
```