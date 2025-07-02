# 1. tmux介绍

<div align=center>
    <img src=./imgs_markdown/2025-07-02-11-11-22.png
    width=25%></br><center></center>
</div>


Tmux 是一款功能强大的终端复用工具，相比 screen 提供更丰富的会话管理和分屏功能。

# 2. tmux安装

```bash
# Ubuntu / Debian
apt-get install tmux

# CentOS / RHEL
yum install tmux

# macOS
brew install tmux
```

# 3. tmux基本操作

| 命令                                      | 描述                         |
| :---------------------------------------- | :--------------------------- |
| `tmux new -s <session_name>`              | 创建新会话                   |
| `tmux ls`                                 | 列出所有会话                 |
| `tmux a`                                  | 恢复上次会话                 |
| `tmux a -t <session_name>`                | 恢复指定会话                 |
| `tmux kill-session -t <session_name>`     | 关闭指定会话                 |
| `tmux rename-session -t 0 <session_name>` | 重命名会话                   |
| <kbd>prefix</kbd> + <kbd>D</kbd>          | 分离会话（后台运行）         |
| <kbd>prefix</kbd>  + <kbd>S</kbd>         | （在会话中）列出所有会话     |
| <kbd>prefix</kbd> + <kbd>[</kbd>          | 进入复制模式                 |
| <kbd>prefix</kbd> + <kbd>]</kbd>          | 粘贴复制的文本               |

⚠️注意事项：
- <kbd>prefix</kbd>是前缀键，tmux默认是<kbd>Ctrl + B</kbd>，在tmux配置文件中可以进行自定义（我使用<kbd>Ctrl + A</kbd>）
- 有关使用<kbd>prefix</kbd>的命令是先按<kbd>prefix</kbd>再按其他键（记得松掉<kbd>Ctrl</kbd>）

# 4. tmux配置

```bash
# 编辑配置文件（若不存在则创建）
vim ~/.tmux.conf
```

添加以下内容：

```bash
# 使用 Ctrl+a 作为前缀键
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# 默认打开bash而非shell
set -g default-command "bash"

# 启动鼠标支持
set -g mouse on

# 美化状态栏
set -g status-fg white
```

# 5. 使用示例

```bash
# 新建会话
tmux new -s <session_name>

# 在tmux窗口运行所需的程序

# 按下prefix + d将会话分离（或者tmux detach）

# 下次使用时，重新连接到会话
tmux attach-session -t <session_name>
```

# 6. 进阶语法

| 快捷键                                     | 功能说明                                             |
| :----------------------------------------- | :--------------------------------------------------- |
| <kbd>prefix</kbd> + <kbd>%</kbd>             | 划分左右两个窗格                                     |
| <kbd>prefix</kbd> + <kbd>"</kbd>             | 划分上下两个窗格                                     |
| <kbd>prefix</kbd> + <kbd>方向键</kbd>        | 光标切换到其他窗格                                   |
| <kbd>prefix</kbd> + <kbd>;</kbd>             | 光标切换到上一个窗格                                 |
| <kbd>prefix</kbd> + <kbd>o</kbd>             | 光标切换到下一个窗格                                 |
| <kbd>prefix</kbd> + <kbd>{</kbd>             | 当前窗格与上一个窗格交换位置                         |
| <kbd>prefix</kbd> + <kbd>}</kbd>             | 当前窗格与下一个窗格交换位置                         |
| <kbd>prefix</kbd> + <kbd>Ctrl+o</kbd>        | 所有窗格向前移动一个位置，第一个窗格变成最后一个窗格 |
| <kbd>prefix</kbd> + <kbd>Alt+o</kbd>         | 所有窗格向后移动一个位置，最后一个窗格变成第一个窗格 |
| <kbd>prefix</kbd> + <kbd>x</kbd>             | 关闭当前窗格                                         |
| <kbd>prefix</kbd> + <kbd>!</kbd>             | 将当前窗格拆分为一个独立窗口                         |
| <kbd>prefix</kbd> + <kbd>z</kbd>             | 当前窗格全屏显示，再使用一次会变回原来大小           |
| <kbd>prefix</kbd> + <kbd>Ctrl + 方向键</kbd> | 按箭头方向调整窗格大小                               |
| <kbd>prefix</kbd> + <kbd>q</kbd>             | 显示窗格编号                                         |

# 7. 参考

1. [https://github.com/tmux/tmux/wiki/Getting-Started](https://github.com/tmux/tmux/wiki/Getting-Started)
2. [Tmux 使用教程](https://www.ruanyifeng.com/blog/2019/10/tmux.html)