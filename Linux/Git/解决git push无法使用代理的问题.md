# 1. 问题描述

我在cmd终端中输入了`set HTTP_PROXY=http://127.0.0.1:7890`和`set HTTPS_PROXY=http://127.0.0.1:7890`（我使用的是Clash进行的代理，且端口号默认为`7890`），并且我也给`git`添加了代理：

```bash
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

在进行上述操作之后，我使用`curl www.google.com`后终端返回了东西。然后我`git add`和`git commit`了一些文件，在使用`git push`后，终端卡住了几秒，之后提示：

```
ssh: connect to host github.com port 22: Connection timed out
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```

# 2. 问题原因

1. **Git 使用了 SSH 协议**：
   - 从错误信息 `ssh: connect to host github.com port 22: Connection timed out` 可以看出，Git 尝试使用 SSH 协议（默认端口 22）连接到 GitHub，但由于超时导致连接失败。
   - 一般情况下，防火墙、代理配置或 ISP 限制会导致端口 22 的连接被阻断。

2. **代理配置仅作用于 HTTP 和 HTTPS**：
   - 我们配置了 `HTTP_PROXY` 和 `HTTPS_PROXY` 环境变量，以及 Git 的 HTTP 和 HTTPS 代理地址，但这些仅对 HTTP/HTTPS 协议有效，不适用于 SSH 协议。
   - 使用 SSH 协议时，代理需要额外的设置。

3. **Clash 的代理未正确作用于 SSH**：
   - 即便我们在 Clash 中配置了代理，SSH 协议默认不会自动走 HTTP/HTTPS 代理，而需要通过特定工具（如 `ProxyCommand`）或改用其他协议（如 HTTPS）。

# 3. 解决方法

如果我们不强制需要使用 SSH，可以将远程仓库 URL 改为 HTTPS 协议。HTTPS 协议支持通过我们已配置的代理正常工作。操作步骤如下：

## 3.1. 为Linux终端添加代理

```bash
vim ~/.bashrc
```

在文件末尾追加内容：

```
# Proxy
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

> ⚠️Clash默认的代理端口为7890，如果你的不是请修改为正确的端口名

之后更新：

```bash
source ~/.bashrc
```

测试是否成功：

```bash
curl www.google.com
```

如果返回一堆东西，说明终端已经可以使用代理了。

> ⚠️ ping命令是不走代理的，所以不要用ping命令去测试，没有意义。具体说明：[windows终端命令行下如何使用代理？ #1489](https://github.com/shadowsocks/shadowsocks-windows/issues/1489)

## 3.2. 查看git连接远程仓库的方式

```bash
# 查看当前远程仓库 URL：
git remote -v
```

```
origin  git@github.com:userName/repoName.git (fetch)
origin  git@github.com:userName/repoName.git (push)
```

## 3.3. 从SSH协议切换为HTTPS协议

如果远程仓库 URL 是类似 `git@github.com:username/repo.git`，说明使用的是 SSH 协议。我们需要修改为 HTTPS 协议（假设我们的仓库地址为 `https://github.com/username/repo.git`）：

```bash
git remote set-url origin https://github.com/username/repo.git
```

设置完成之后我们尝试 `git push`：

```bash
git push
```

此时应该会提示输入账号和密码，那我们可以<kbd>Ctrl + C</kbd>打断了，因为我们原始的密码无法正常登录的。

## 3.4. 使用 GitHub Personal Access Token (PAT) 登录

### 3.4.1. 创建 Personal Access Token

1. 登录到我们的 GitHub 账户：[https://github.com/login](https://github.com/login)
2. 点击右上角头像，选择 **Settings**（设置）。
3. 在左侧栏中找到 **Developer settings**（开发者设置）。
4. 点击 **Personal access tokens > Tokens (classic)**。
5. 点击 **Generate new token**（生成新令牌），然后选择 **Generate new token (classic)**。
6. 在生成令牌的页面：
   - **Note**: 添加一个描述，比如 "Git Push Token"。
   - **Expiration**: 选择一个合适的过期时间（建议选择最长的 "No expiration" 如果我们不想频繁更新令牌）。
   - **Scopes**: 勾选以下权限：
     - `repo`（访问私有和公共仓库）。
   - 点击 **Generate token**。

7. 生成后，GitHub 会显示我们的 Personal Access Token（PAT）。**立即复制保存**，<font color='red'><b>因为之后将无法再次查看</b></font>。

> 💡不保存也可以的，如果还需要用我们再次生成也可以（但会导致之前的失效）

### 3.4.2. 配置 Git 使用 PAT

当我们使用 HTTPS 推送代码时，需要用 Personal Access Token 替代密码。

1. 再次尝试 `git push`，当提示输入用户名和密码时：
   - **Username**: 输入我们的 GitHub 用户名。
   - **Password**: 输入刚刚生成的 Personal Access Token。

如果推送成功，则配置完成。

### 3.4.3. 保存 PAT（可选）

为了避免每次推送都需要手动输入 PAT，我们可以将其保存到 Git 的凭据管理器中。

#### 3.4.3.1. 方法 1：通过 Git 自带的凭据管理器保存

运行以下命令启用凭据缓存：
```bash
git config --global credential.helper store
```

然后再次运行 `git push`，输入用户名和 PAT 后，Git 会将凭据保存到本地。在下一次推送时，不会再次提示输入。

#### 3.4.3.2. 方法 2：手动编辑 `.gitconfig`

编辑我们的全局 Git 配置文件：

```bash
vim ~/.gitconfig
```

添加以下内容（将 `USERNAME` 替换为我们的 GitHub 用户名，将 `TOKEN` 替换为我们的 Personal Access Token）：

```
[credential "https://github.com"]
    username = USERNAME
    helper = store
```

保存后，Git 会自动使用保存的用户名和 PAT。

# 4. 其他方法

> ⚠️<font color='red'><b>推荐使用上面的方法，已经亲自测试过，没有什么问题</b></font>。

## 4.1. 方法2：配置 SSH 通过代理

如果我们必须使用 SSH 协议，可以配置 SSH 通过代理连接。

**操作步骤**：
- 编辑 SSH 配置文件：
    ```bash
    vim ~/.ssh/config
    ```
- 添加以下内容：
    ```plaintext
    Host github.com
        HostName github.com
        User git
        ProxyCommand nc -X 5 -x 127.0.0.1:7890 %h %p
    ```
    解释：
    - `ProxyCommand` 指定通过代理访问 GitHub。
    - `nc` 是 `netcat` 的缩写，用于通过代理转发请求。
    - `-X 5` 指定代理类型为 SOCKS5。
    - `-x 127.0.0.1:7890` 指定代理地址为 Clash 的本地代理。

- 保存配置文件后，重试 `git push`。

**注意**：
- 确保我们的 Clash 已正确配置为全局模式或开启了 SOCKS5 代理功能。
- `nc` 需要提前安装，如果未安装可以参考相关教程。


## 4.2. 方法3：检查网络环境

如果上述方法仍然无效，可能我们的网络环境存在更深层次的问题（如防火墙限制或 Clash 配置问题）。可以尝试以下操作：
- 确认 Clash 的代理是否正常工作：
    ```bash
    curl -x socks5h://127.0.0.1:7890 https://www.google.com
    ```
    如果返回内容正常，说明代理工作正常。
- 确保 Clash 的规则允许 GitHub 的 SSH 或 HTTPS 流量通过。
- 检查是否有其他网络限制（如公司网络屏蔽端口）。

## 4.3. 方法4：使用 GitHub CLI 登录并推送

如果 HTTPS 和 SSH 都无法解决问题，可以使用 [GitHub CLI](https://cli.github.com/) 进行身份验证和推送。
1. 安装 GitHub CLI。
2. 登录：
    ```bash
    gh auth login
    ```
3. 选择 HTTPS 协议并登录。
4. 再次尝试推送代码。

## 4.4. 方法5：使用其他端口（非 22 端口的 SSH）

GitHub 提供了一个备用的 SSH 端口（443），可以尝试使用该端口：

- 编辑 SSH 配置文件：
    ```bash
    notepad %USERPROFILE%\.ssh\config
    ```
- 添加以下内容：
    ```plaintext
    Host github.com
        HostName ssh.github.com
        User git
        Port 443
        ProxyCommand nc -X 5 -x 127.0.0.1:7890 %h %p
    ```
- 保存后重试 `git push`。

# 5. 总结
- **推荐优先方案**：切换到 HTTPS 协议，简单且无需额外配置。
- 如果必须使用 SSH，配置 SSH 的代理或切换到 GitHub 的备用端口（443）。
- 确保 Clash 的代理规则和网络环境无额外限制。

# 知识来源
- [DeepSeek](https://chat.deepseek.com/)
- [ChatGPT-4o](https://lmarena.ai/)