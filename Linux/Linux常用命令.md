💡 建议直接 <kbd>Ctrl + F</kbd> 搜索使用

### 解压缩.tar.gz文件

```bash
[1]解压文件到当前目录：tar -xvf xxx.tar.gz
[2]解压文件到指定目录（如果指定目录不存在则会报错）：tar -xvf xxx.tar.gz -C <target_dir>
```

###  查看文件/文件夹大小

```bash
# 查看某个文件大小
du -h filepath

# 查看某个文件夹总大小
du -sh dirpath

# 查看某个文件夹所有文件的大小
du -ah dirpath
```

### 查看文件/文件夹数量

```bash
# 查看当前文件夹下文件数量
ls -l | wc -l

# 查看指定文件夹下文件数量
find 文件夹路径 -type f | wc -l

# 文件
find xxxx/xxx -max_depth n -type f | wc -l

# 文件夹
find xxxx/xxx -max_depth n -type d | wc -l

# 某种类型的数量（引号必须要有）
find xxxx/xxx -max_depth n -type f -name '*.txt' | wc -l
```

### 7z 相关

```bash
# 安装7zip
apt-get install p7zip-full

# 压缩文件
7z a 压缩包名称.格式 file1 file2 file3

# 解压文件（⚠️注意-o和目标路径是连起来的，没有空格）
7z x 压缩包名称.格式 -o解压的路径
```

### conda 相关

```bash
# 查看conda拥有的环境
conda env list

# 创建conda环境
conda create -n 环境名称 python=3.8

# 克隆conda环境
conda create -n 新环境名称 --clone 被克隆的环境名称

# 删除conda环境
conda remove -n 环境名称 --all
```

### 安装软件

```bash
# 更新软件包
apt-get update

# 安装软件
apt-get install 软件名
```

### 查看文档

```bash
# 查看文档内容
cat filename

# 编辑文档
vim filename

# 在WSL中使用记事本打开文档
notepad.exe filename
```

### vim 相关

```bash
# vim相关

# 使用vim打开文档
vim filename

# 编写
i

# 保存并退出
ESC + :wq

# 仅退出
ESC + :q
```

### screen 相关

```bash
# screen 相关

# 安装screen
apt-get install screen

# 创建session
screen -U -R 会话名称

# 进入已创建好的session
screen -U -r 会话名称

# 查看当前有哪些session被创建
screen -ls -U

# 关闭当前session
ctrl+A+D

# 查看终端中内容
ctrl + [
```

### 服务器传输文件

```bash
# 使用 scp 
scp -r -P 端口号 要复制的文件夹 目标服务器用户名@目标服务器地址:目标服务器文件夹

# 使用 rsync
apt-get install rsync  # 两个服务器都需要安装
rsync -r -P --rsh='ssh -p 目标服务器端口' 要复制的文件夹 目标服务器用户名@目标服务器地址:目标服务器文件夹
```

### 修改容器密码

```bash
passwd
```