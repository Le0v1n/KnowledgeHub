💡 建议直接 <kbd>Ctrl + F</kbd> 搜索使用

# 内置命令相关

## 查看当前路径

```bash
pwd
```

## 查看文档

```bash
# 查看文档内容
cat filename

# 编辑文档
vim filename

# 在WSL中使用记事本打开文档
notepad.exe filename
```

## 查看文件/文件夹大小

```bash
# 查看某个文件大小
du -h filepath

# 查看某个文件夹总大小
du -sh dirpath

# 查看某个文件夹所有文件的大小
du -ah dirpath
```

## 查看文件/文件夹数量

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

# 硬件相关

## 查看服务器信息

```bash
# 查看CPU
lscpu

# 查看内存信息
free -h

# 查看显卡
nvidia-smi
nvidia-smi -L
```

# 解压/压缩相关

## tar

```bash
# 解压缩.tar.gz文件

# 解压文件到当前目录
tar -xvf xxx.tar.gz

# 解压文件到指定目录（如果指定目录不存在则会报错）
tar -xvf xxx.tar.gz -C <target_dir>
```

## 7z

```bash
# 安装7zip
apt-get install p7zip-full

# 压缩文件
7z a 压缩包名称.格式 file1 file2 file3

# 解压文件（⚠️注意-o和目标路径是连起来的，没有空格）
7z x 压缩包名称.格式 -o解压的路径
```

# Python相关

## 环境管理

### Anaconda

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

### venv

> ⚠️虚拟环境会直接创建在`pwd`目录中，所以我们需要提前`cd`到指定目录中以便管理虚拟环境

```bash
# 创建虚拟环境（虚拟环境会直接创建在pwd目录中，所以我们需要提前cd到指定目录中以便管理虚拟环境）
cd <虚拟环境创建的文件夹中>
python -m venv <虚拟环境名称>

# 查找拥有的虚拟环境
find . -type f -name "activate"

# 激活虚拟环境
source <虚拟环境名称>/bin/activate

# 退出虚拟环境
deactivate

# 删除虚拟环境
rm -rf <虚拟环境名称>
```

## 查找当前Python的位置

```bash
which python
```

# 安装软件相关

```bash
# 更新软件包
apt-get update

# 安装软件
apt-get install 软件名
```

# vim相关

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

# screen相关

```bash
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

# 传输文件/文件夹相关

## 服务器传输文件

```bash
# 使用 scp 
scp -r -P 端口号 要复制的文件夹 目标服务器用户名@目标服务器地址:目标服务器文件夹

# 使用 rsync
apt-get install rsync  # 两个服务器都需要安装
rsync -r -P --rsh='ssh -p 目标服务器端口' 要复制的文件夹 目标服务器用户名@目标服务器地址:目标服务器文件夹
```

# docker相关

## 修改容器密码

```bash
passwd
```

# Git相关

## 1. 账号相关

### 1. 生成SSH-Key

```bash
# 查看ssh目录是否存在
cd ~/.ssh

# 如果不存在则需要创建（将 "xxx@xxx.com" 替换为你自己GitHub的邮箱地址）
ssh-keygen -t rsa -C "xxx@xxx.com"

# 查看SSH-Key公钥信息
cd ~/.ssh
cat id_rsa.pub

# 复制公钥信息

# 在GitHub中添加公钥，URL为：https://github.com/settings/keys
```

### 2. 克隆仓库

```bash
git clone <ssh链接>
```

### 3. 创建账户

在`git commit`的时候可能会碰到问题 “please tell me who you are”，输入下面的命令：

```bash
# "you@example.com"替换为你的GitHub邮箱地址
git config --global user.email "you@example.com"

# "Your Name"替换为你的GitHub名称
git config --global user.name "Your Name"
```

### 4. git push需要输入账号密码

运行以下命令将远程仓库地址更新为 SSH 地址：

```bash
git remote set-url origin git@github.com:<用户名>/<仓库名>.git
```

## 2. 命令

### 1. branch分支相关

```bash
# 列出所有分支
git branch

# 创建新的分支
git branch <branch_name>

# 删除某个分支
git branch -d <branch_name>

# 切换到某个分支
git checkout <branch_name>
```

### 2. 提交代码

```bash
# 将文件或文件夹添加到暂存区
git add filepath/dirpath
git add .

# 将修改放到本地仓库
git commit -m "your useful commit message"

# 提交代码
git push
```