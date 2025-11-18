# 1. LangChain的介绍

## 1.1. Langchain是什么？

想象一下，一个聊天机器人不仅能回答常见问题，还能从我们专属的数据库或文件中提取信息，并基于这些信息执行具体操作，例如发送电子邮件。这不仅是未来的愿景，而是现在可以实现的现实。

为此，LangChain 应运而生。LangChain 是一个开源框架，专门设计用于将大型语言模型（如 GPT-4、ChatGLM等）与外部计算和数据源结合。通过这一框架，开发者可以创建由语言模型驱动的强大应用程序。

LangChain 的核心在于它通过以下三个主要组件来增强大型语言模型的功能：

1. **Components（组件）**：这些组件为大型语言模型提供了接口封装、模板提示以及信息检索索引。它们是与模型交互的基础设施，使语言模型能够理解并处理复杂的任务。
   
2. **Chains（链）**：Chains 是将不同组件组合在一起以解决特定任务的流程。通过链的组合，开发者可以让模型在大量文本中查找信息，或执行更复杂的逻辑操作。
   
3. **Agents（代理）**：代理允许大型语言模型与外部环境进行交互。这意味着模型不仅能理解和生成文本，还能通过 API 请求执行操作，如访问数据库、调用外部服务等。

通过 LangChain，开发者能够赋予语言模型以更高的自主性和功能性，使其能够处理更复杂和动态的任务。这种能力的提升，使得 AI 应用的开发变得更加灵活和强大。

> 🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：LangChain可以训练、微调LLM吗？
>
> 🥳 𝑨𝒏𝒔𝒘𝒆𝒓：LangChain <font color='red'><b>不直接支持训练或微调大型语言模型（LLM）</b></font>，LangChain 更多是作为一个应用开发框架，专注于将 LLM 集成到实际应用中，而不是直接用于训练或微调模型。
>
> 😐如果需要进行模型的训练或微调，建议结合其他专门的工具（如 Hugging Face Transformers 或 PyTorch）来实现。

## 1.2. 为什么要用 LangChain？

- **数据连接**：LangChain 让我们可以把大型语言模型连接到自己的数据源，比如数据库、PDF 文件或其他文档。这意味着模型可以从我们的私人数据中提取信息，变得更为智能。

- **行动执行**：不仅能获取信息，LangChain 还能帮我们根据这些信息执行具体操作，比如发邮件。这样一来，模型不仅能回答问题，还能实际帮我们做事。

## 1.3. LangChain的组成

- **`langchain-core`**：基本抽象和 LangChain 表达式语言。
- `langchain-community`：第三方集成。
	- 合作伙伴包（例如 **`langchain-openai`**，**`langchain-anthropic`** 等）：某些集成已进一步拆分为仅依赖于 **`langchain-core`** 的轻量级包。
- **`langchain`**：构成应用程序认知架构的链条、代理和检索策略。
- **[langgraph](https://python.langchain.com/v0.2/docs/langgraph)**：通过将步骤建模为图中的边缘和节点，使用LLMs构建强大且有状态的多角色应用程序。
- **[langserve](https://python.langchain.com/v0.2/docs/langserve/)**：将 LangChain 链条部署为 REST API。

## 1.4. LangChain 的核心

- **Models（模型）**：LangChain 提供了包装器，让我们可以轻松连接到大型语言模型，比如 GPT-4 或 Hugging Face 的模型，甚至 GLM。

- **Prompt Templates（提示词模板）**：这些模板帮我们避免硬编码输入。我们可以动态插入用户输入，生成更灵活的提示词，然后发给语言模型。

- **Chains（链）**：链就像把不同的组件串在一起，解决特定任务的流程，让我们能构建完整的语言模型应用。

- **Agents（代理/智能体）**：代理让语言模型可以与外部 API 互动，这样模型不仅能理解信息，还能用这些信息去做实际的操作。

- **Embedding（嵌入）与 VectorStore（向量存储）**：这些技术是用来表示和检索数据的，给模型提供必要的语言理解基础。

- **Indexes（索引）**：索引用来帮我们从语言模型中提取相关信息，让我们的信息检索更高效。

## 1.5. LangChain的底层原理

<div align=center>
    <img src=./imgs_markdown/2025-02-20-20-11-20.png
    width=80%></br><center></center>
</div>

如上图所示展示了一个智能问答系统的工作流程。

1. 从用户提出的问题（Question）开始，然后通过相似性搜索（Similarity Search）在一个大型数据库或向量空间中找到与之相关的信息。
2. 得到的信息与原始问题结合后，由一个处理模型分析，以产生一个答案（Answer）。
3. 这个答案接着被用来指导一个代理采取行动（Action），这个代理可能会执行一个API调用或与外部系统交互以完成任务。

整个流程反映了数据驱动的决策过程，其中包含了从信息检索到处理，再到最终行动的自动化步骤。

## 1.6. LangChain的应用场景

LangChain 的应用场景非常广泛，包括但不限于：

- **个人助手**：可以帮助预订航班、转账、缴税等。
- **学习辅助**：可以参考整个课程大纲，帮助我们更快地学习材料。
- **数据分析和数据科学**：连接到公司的客户数据或市场数据，极大地促进数据分析的进展。

# 2. LangChain实战操作

## 2.1. LangChain的安装

> 官方英文文档：[https://python.langchain.com/docs/introduction/](https://python.langchain.com/docs/introduction/)
>
> 中文文档（非官方）：[https://www.langchain.asia/get_started/introduction](https://www.langchain.asia/get_started/introduction)

```bash
# LangChain本体
pip install langchain

# LangChain专为OpenAI的版本（可以不下载）
pip install langchain-openai
```

## 2.2. LangSmith

LangSmith是一个用于构建生产级 LLM 应用程序的平台，它提供了调试、测试、评估和监控基于任何 LLM 框架构建的链和智能代理的功能，并能与 LangChain 无缝集成。其主要作用包括：

- **调试与测试**：通过记录langchain构建的大模型应用的中间过程，开发者可以更好地调整提示词等中间过程，优化模型响应。
- **评估应用效果**：langsmith可以量化评估基于大模型的系统的效果，帮助开发者发现潜在问题并进行优化。
- **监控应用性能**：实时监控应用程序的运行情况，及时发现异常和错误，确保其稳定性和可靠性。
- **数据管理与分析**：对大语言模型此次的运行的输入与输出进行存储和分析，以便开发者更好地理解模型行为和优化应用。
- **团队协作**：支持团队成员之间的协作，方便共享和讨论提示模板等。
- **可扩展性与维护性**：设计时考虑了应用程序的可扩展性和长期维护，允许开发者构建可成长的系统。

LangSmith是LangChain的一个子产品，是一个大模型应用开发平台。它提供了从原型到生产的全流程工具和服务，帮助开发者构建、测试、评估和监控基于LangChain或其他 LLM 框架的应用程序。

## 2.3. 获取和设置API KEY

### 2.3.1. 获取API Key

#### 2.3.1.1. LangSmith

获取LangSmith的API：[https://smith.langchain.com/settings](https://smith.langchain.com/settings)

#### 2.3.1.2. OpenAI

因为OpenAI会限制地区，所以我这里使用的是国内的网站：https://www.dmxapi.com/register?aff=Te4K

### 2.3.2. 设置API Key

我们可以在终端中声明我们获取得到的LangSmith和OpenAI的API Key。

```bash
# 声明环境变量
export LANGCHAIN_API_KEY="langsmith的API"
export OPENAI_API_KEY="OpenAI的API Key"
```

## 2.4. 尝试调用ChatGPT

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 定义需要传入的提示词
msg = [
    SystemMessage(content="请将以下内容翻译为英语"),
    HumanMessage(content='我们好，请问我们要去哪里？')
]

result = model.invoke(input=msg)

print(result)
```

```
content='Hello, where are you going?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 34, 'total_tokens': 41, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-7a7d88e0-c63f-4da6-ba67-3e43bda36100-0' usage_metadata={'input_tokens': 34, 'output_tokens': 7, 'total_tokens': 41, 'input_token_details': {}, 'output_token_details': {}}
```

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：为什么`msg`要有`SystemMessage`，不应该只要一个`HumanMessage`就可以了吗？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在使用 LangChain 和 OpenAI 的聊天模型时，消息上下文对于生成合理的响应非常重要。我们的代码中包含 `SystemMessage` 和 `HumanMessage`，这是为了更好地控制和指导模型的行为。下面是解释为什么需要 `SystemMessage` 的一些原因：

1. **提供上下文和指令**：`SystemMessage` 通常用于提供模型的初始上下文或指令。这些消息可以帮助模型理解它的角色和任务。例如，我们的 `SystemMessage` 指示模型将内容翻译为英语，这是一种明确的指令，它定义了模型的行为和目标。
   
2. **控制模型的行为**：系统消息可以用来设置模型的语气、风格或特定任务。这对于多轮对话或需要特定行为的任务特别有用。例如，我们可以通过系统消息告诉模型它是一个帮助用户翻译的助手，而不是聊天机器人。
   
3. **确保一致性**：在多轮对话中，`SystemMessage` 可以帮助保持对话的一致性和连贯性。它确保每次对话开始时，模型都知道需要遵循的规则和目标。
   
4. **增强可控性**：带有系统消息的对话通常更具可控性，因为我们可以明确设置期望的输出类型和质量。这对于需要精确度的应用场景（如翻译、技术支持等）非常重要。

因此，虽然在某些简单的场景下，我们可能只需要 `HumanMessage`，但为了确保模型按照预期执行任务，尤其是涉及到明确的操作指令时，`SystemMessage` 是非常有用的。它不仅提升了模型的功能性，还增强了对输出结果的可预测性和一致性。

那我们尝试一下将`SystemMessage`删掉再看看结果：

```
content='我们好！我只是一个虚拟助手，无法去任何地方。请问有什么可以帮到您的吗？' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 18, 'total_tokens': 56, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-f26a39b3-2ade-4079-b05e-e955421d33b6-0' usage_metadata={'input_tokens': 18, 'output_tokens': 38, 'total_tokens': 56, 'input_token_details': {}, 'output_token_details': {}}
```

我们可以发现回复的结果也变了。

## 2.5. 解析结果

我们现在来看一下LLM的回复：

```
content='Hello, where are you going?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 34, 'total_tokens': 41, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-7a7d88e0-c63f-4da6-ba67-3e43bda36100-0' usage_metadata={'input_tokens': 34, 'output_tokens': 7, 'total_tokens': 41, 'input_token_details': {}, 'output_token_details': {}}
```

这段输出包含了从调用 LangChain 时返回的详细信息，涉及生成的内容、元数据和使用情况统计。下面是对这段输出的分析：

1. **内容（content）**：
   - `content='Hello, where are you going?'`：这是模型生成的文本内容。根据我们的输入，模型将中文翻译成了英语。

2. **附加关键字参数（additional_kwargs）**：
   - `additional_kwargs={'refusal': None}`：这通常是用来处理模型拒绝响应的情况。在这个例子中，它是 `None`，表示模型没有拒绝请求。

3. **响应元数据（response_metadata）**：
   - `response_metadata`：这是关于模型响应的一些元数据信息。
     - `token_usage`：显示了令牌的使用情况。
       - `completion_tokens`: 7 表示生成的响应使用了 7 个令牌。
       - `prompt_tokens`: 34 表示提示信息（包括系统消息和用户消息）使用了 34 个令牌。
       - `total_tokens`: 41 表示总共使用了 41 个令牌。
     - `model_name`: 'gpt-3.5-turbo-0125' 表示使用的模型版本。
     - `system_fingerprint`: 'fp_0165350fbb' 是与系统相关的一个标识符。
     - `finish_reason`: 'stop' 表示生成的结束原因，<font color='red'><b>这里是因为模型自然结束，而不是被截断</b></font>。
     - `logprobs`: None 表示没有返回概率日志信息。

4. **标识符（id）**：
   - `id='run-7a7d88e0-c63f-4da6-ba67-3e43bda36100-0'`：这是这次模型调用的唯一标识符，可以用于跟踪或记录这次运行。

5. **使用情况元数据（usage_metadata）**：
   - `usage_metadata`：提供了有关输入和输出令牌的详细使用情况。
     - `input_tokens`: 34 表示输入使用了 34 个令牌。
     - `output_tokens`: 7 表示输出使用了 7 个令牌。
     - `total_tokens`: 41 表示总令牌数。
     - `input_token_details` 和 `output_token_details` 是空字典，表明没有进一步的令牌详细信息。

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：在`token_usage`字段中，`completion_tokens`表示LLM回复花费的token数量吗？`prompt_tokens`表示`SystemMessage`个`HumanMessage`的token数量吗？在`usage_metadata`字段中，`input_tokens`表示`SystemMessage`个`HumanMessage`的token数量吗？`output_tokens`表示LLM回复花费的token数量吗？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：我们对 `token_usage` 和 `usage_metadata` 字段的理解大体是正确的。以下是对这些字段的详细解释：

1. **`token_usage` 字段**：
   - **`completion_tokens`**：表示生成的回复所花费的令牌数量。这是模型在生成输出时使用的令牌数。
   - **`prompt_tokens`**：表示输入部分（包括 `SystemMessage` 和 `HumanMessage`）所花费的令牌数量。这是模型在处理输入时使用的令牌数。

2. **`usage_metadata` 字段**：
   - **`input_tokens`**：与 `prompt_tokens` 类似，表示输入部分（包括 `SystemMessage` 和 `HumanMessage`）的令牌数量。
   - **`output_tokens`**：与 `completion_tokens` 类似，表示生成的回复所花费的令牌数量。

简而言之，`prompt_tokens` 和 `input_tokens` 都指的是输入消息（系统消息和用户消息）的令牌计数，而 `completion_tokens` 和 `output_tokens` 都指的是模型生成的输出的令牌计数。这些计数对于了解模型的使用情况，尤其是在计费和优化时非常重要。

## 2.6. 返回数据解析器

```
content='Hello, where are you going?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 34, 'total_tokens': 41, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-7a7d88e0-c63f-4da6-ba67-3e43bda36100-0' usage_metadata={'input_tokens': 34, 'output_tokens': 7, 'total_tokens': 41, 'input_token_details': {}, 'output_token_details': {}}
```

这是LLM的回复，我们发现其实如果给用户使用的话很多内容都是冗余的，因此我们需要对LLM的回复进行解析，得到我们想要的即可：

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 定义需要传入的提示词
msg = [
    SystemMessage(content="请将以下内容翻译为英语"),
    HumanMessage(content='我们好，请问我们要去哪里？')
]

result = model.invoke(input=msg)

print(f"[原始结果] {result}\n")

# 创建解析器
parser = StrOutputParser()
print(f"[解析后的结果] {parser.invoke(result)}")
```

```
[原始结果] content='Hello, where are you going?' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 34, 'total_tokens': 41, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_0165350fbb', 'finish_reason': 'stop', 'logprobs': None} id='run-2bd43e6d-e334-40a9-8c54-ba1d1a7a0aca-0' usage_metadata={'input_tokens': 34, 'output_tokens': 7, 'total_tokens': 41, 'input_token_details': {}, 'output_token_details': {}}

[解析后的结果] Hello, where are you going?
```

## 2.7. 构建链chain

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：有人可能问了：“主包主包，先创建模型再创建解析器去解析还是太难了，有没有更好的方法？”

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：有的有的。LangChain重载了`|`操作符，那么我们就可以这样操作了：

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 定义需要传入的提示词
msg = [
    SystemMessage(content="请将以下内容翻译为英语"),
    HumanMessage(content='我们好，请问我们要去哪里？')
]

# 构建解析器
parser = StrOutputParser()

# 创建链
chain = model | parser

# 直接使用chain来调用
print(f"[Chain] {chain.invoke(msg)}")
```

```
[Chain] Hi, may I ask where are you going?
```

# 3. LangChain的提示模板（PromptTemplate）

## 3.1. 代码示例

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 构建解析器
parser = StrOutputParser()

# 定义PromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', '将下面内容翻译为{language}'),
        ('user', '{text}')
    ]
)

# 创建链
chain = prompt_template | model | parser

# 直接使用chain来调用
print(chain.invoke(
    {
        'language': '日语',
        'text': '见到我们很高兴!'
    }
))
```

```
あなたに会えて嬉しいです！
```

其实看到这里大家可能对其中的语法` ('system', '将下面内容翻译为{language}')`比较陌生，我也陌生😂。下面我们介绍一下这种语法。

## 3.2. LangChain `ChatPromptTemplate.from_messages()` 语法详解

在使用 **LangChain** 进行 AI 对话任务时，Prompt 设计是至关重要的一环。LangChain 提供了 **`ChatPromptTemplate.from_messages()`** 方法，允许开发者以结构化的方式定义 Prompt，使其更符合对话模型的输入格式。

### 3.2.1. `from_messages()` 语法介绍

在 LangChain 中，`ChatPromptTemplate.from_messages()` 允许开发者使用 **列表（List）** 的方式定义 Prompt，其中每个元素都是一个 **消息 (Message)**，格式如下：

```python
("角色", "消息内容")
```

**角色 (Role)**有三种：

1. `"system"`：系统消息，通常用于设定 AI 的行为、风格或背景信息。
2. `"user"`：用户消息，表示用户的输入内容。
3. `"assistant"`（可选）：AI 预设的回复，可用于提供示例对话。


### 3.2.2. 语法示例

### 3.2.3. 示例 1：基本用法
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "我们是一个翻译助手"),
    ("user", "请将以下内容翻译成 {language}: {text}")
])

formatted_prompt = prompt.format(language="法语", text="我们好")
print(formatted_prompt)
```

```
System: 我们是一个翻译助手
User: 请将以下内容翻译成 法语: 我们好
```

- `("system", "...")` 设置 AI 的角色，让 AI 知道它是一个翻译助手。
- `("user", "...")` 定义用户输入，`{language}` 和 `{text}` 是占位符，运行时会替换成具体值。


#### 3.2.3.1. 示例 2：包含 AI 预设回复

有时，我们希望提供一些示例对话，让 AI 更好地理解任务。这时可以使用 **`("assistant", "...")`** 角色。

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "我们是一名专业的英语翻译员"),
    ("user", "请将以下句子翻译成英文: {text}"),
    ("assistant", "好的，这句话的英文翻译是：")
])

formatted_prompt = prompt.format(text="今天天气很好")
print(formatted_prompt)
```

```
System: 我们是一名专业的英语翻译员
User: 请将以下句子翻译成英文: 今天天气很好
Assistant: 好的，这句话的英文翻译是：
```

```
好的，这句话的英文翻译是：I'm glad to see you!
```

- `("assistant", "...")` 让 AI 以固定格式回复，有助于引导 AI 输出更符合预期的内容。


#### 3.2.3.2. 示例 3：结合 LangChain LLM 调用

在实际使用中，我们通常会将 `ChatPromptTemplate` 与 **LangChain LLM** 结合，创建完整的 AI 交互链。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 构建解析器
parser = StrOutputParser()

# 定义PromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "我们是一名专业的英语翻译员"),
    ("user", "请将以下句子翻译成英文: {text}"),
    ("assistant", "好的，这句话的英文翻译是：")
])

# 创建链
chain = prompt | model | parser

# 直接使用chain来调用
print(chain.invoke(
    {
        'text': '见到我们很高兴!'
    }
))
```


### 3.2.4. `from_messages()` 的优势

- **结构化对话**：比传统的 `PromptTemplate` 更适合多轮对话场景。
- **上下文管理**：清晰区分 **系统指令**、**用户输入** 和 **AI 预设回复**，更易维护。
- **模板化填充**：支持 `{变量}` 语法，方便动态填充内容。
- **兼容 LangChain Pipeline**：可以直接与 LLM 结合，形成完整的 AI 交互。

# 4. 将LLM封装为一个服务

## 4.1. 安装库

```bash
pip install langserve[all]
```

## 4.2. 案例一：翻译器

### 4.2.1. 代码

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 调用LLM
model = ChatOpenAI(
    model='gpt-3.5-turbo',
    base_url="https://www.dmxapi.com/v1"
)

# 构建解析器
parser = StrOutputParser()

# 定义PromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', '将下面内容翻译为{language}'),
        ('user', '{text}')
    ]
)

# 创建链
chain = prompt_template | model | parser

# 直接使用chain来调用
# print(chain.invoke(
#     {
#         'language': '英语',
#         'text': '见到你很高兴!'
#     }
# ))

# ========== 将程序部署为服务 ==========
# 创建FastAPI应用
app = FastAPI(
    title='My LangChain Server',
    version='v1.0',  # 服务的版本（我们自定义的）
    description='使用LangChain的翻译服务'
)

# 添加路由
add_routes(
    app,
    chain,
    path='/chainDemo'  # 请求路径
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有 IP
        port=8080
    )
```

```
INFO:     Started server process [432103]
INFO:     Waiting for application startup.

     __          ___      .__   __.   _______      _______. _______ .______     ____    ____  _______
    |  |        /   \     |  \ |  |  /  _____|    /       ||   ____||   _  \    \   \  /   / |   ____|
    |  |       /  ^  \    |   \|  | |  |  __     |   (----`|  |__   |  |_)  |    \   \/   /  |  |__
    |  |      /  /_\  \   |  . `  | |  | |_ |     \   \    |   __|  |      /      \      /   |   __|
    |  `----./  _____  \  |  |\   | |  |__| | .----)   |   |  |____ |  |\  \----.  \    /    |  |____
    |_______/__/     \__\ |__| \__|  \______| |_______/    |_______|| _| `._____|   \__/     |_______|
    
LANGSERVE: Playground for chain "/chainDemo/" is live at:
LANGSERVE:  │
LANGSERVE:  └──> /chainDemo/playground/
LANGSERVE:
LANGSERVE: See all available routes at /docs/
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### 4.2.2. 使用LangServe的网页调用

进入我们的网址：[http://127.0.0.1:8080/chainDemo/playground/](http://127.0.0.1:8080/chainDemo/playground/)

<div align=center>
    <img src=./imgs_markdown/2025-02-27-11-40-12.png
    width=80%></br><center></center>
</div>

### 4.2.3. 使用Python调用

```python
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
```

```
result = 'Nice to meet you!'
```

### 4.2.4. 使用ApiPost软件调用

<div align=center>
    <img src=./imgs_markdown/2025-02-27-09-56-25.png
    width=100%></br><center></center>
</div>

> 数据格式要设置为json

### 4.2.5. FAQ

#### 4.2.5.1. playground界面

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：`http://127.0.0.1:8080/chainDemo/playground/`这个playground页面是uvicorn、FastAPI、LangServer中的谁提供的？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：这个 **"Playground" 页面** 不是 `Uvicorn` 或 `FastAPI` 提供的，而是 **`LangServe`** 提供的。  

<font color='blue'><b>【Uvicorn】</b></font>

`Uvicorn` 只是一个 **ASGI 服务器**，它的作用是运行 FastAPI 应用，并监听 HTTP 请求。  
- **Uvicorn 本身不会提供 Web UI（如 Playground）**，它只是一个服务器。  
- 我们可以用 `uvicorn.run(...)` 或 `uvicorn app:app` 启动 FastAPI，但它不会提供 `Playground` 页面。  

**✅ 结论：** Uvicorn **不会** 生成 `Playground` 页面。

<font color='blue'><b>【FastAPI】</b></font>

`FastAPI` 默认提供 **Swagger UI (`/docs`)** 和 **ReDoc (`/redoc`)**，但它 **不会提供 `Playground` 页面**。  
- 访问 `http://127.0.0.1:8080/docs` 可以打开 **Swagger UI**（FastAPI 内置）。  
- 访问 `http://127.0.0.1:8080/redoc` 可以打开 **ReDoc 文档**（FastAPI 内置）。  

**✅ 结论：** FastAPI **不会** 生成 `Playground` 页面，它只提供 `Swagger UI` 和 `ReDoc`。

<font color='blue'><b>【LangServe】</b></font>

我们在代码中使用了 `langserve.add_routes()`：

```python
from langserve import add_routes

# 添加 LangChain 作为 API 路由
add_routes(
    app,
    chain,
    path='/chainDemo'
)
```

- `LangServe` 是 `LangChain` 提供的一个工具，用于将 `LangChain` 的 `Chain` 变成 REST API。  
- **LangServe 内置了一个 `Playground` 页面**，让我们可以直接在浏览器里测试 `LangChain Chain`。  
- 这个 `Playground` 页面通常位于：
  ```
  http://127.0.0.1:8080/chainDemo/playground/
  ```

<font color='blue'><b>【结论】</b></font>

`"Playground"` 页面是 **`LangServe` 提供的**，用于测试 LangChain 的 `Chain`。

| 组件      | 是否提供 `Playground` | 说明                                                                   |
| --------- | --------------------- | ---------------------------------------------------------------------- |
| Uvicorn   | ❌ 否                  | 只是 ASGI 服务器，不提供 UI                                            |
| FastAPI   | ❌ 否                  | 提供 `Swagger UI` (`/docs`) 和 `ReDoc` (`/redoc`)，但没有 `Playground` |
| LangServe | ✅ 是                  | **提供 `/chainDemo/playground/` 页面**，用于测试 `LangChain`           |

#### 4.2.5.2. IP相关

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：我使用的是wsl2，但我为什么无法使用ApiPost软件发送POST情况呢，返回了502的状态码？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：需要设置一下端口转发，操作步骤如下：

```bash
# 在PowerShell（⚠️管理员模式）

# 添加端口映射
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=<wsl2的ip>

# Windows 防火墙可能会拦截转发的端口，因此需要添加一个规则
netsh advfirewall firewall add rule name="WSL2 8080" dir=in action=allow protocol=TCP localport=8080

# 查看当前映射了哪些
netsh interface portproxy show all
```

> 可以在wsl2的终端输入`ifconfig`查看wsl2的ip

正常来说应该显示：

```
地址            端口        地址            端口
--------------- ----------  --------------- ----------
0.0.0.0         8080        <wsl2的ip地址>   8080
```

之后我们可以尝试在Windows浏览器访问：

```html
http://10.72.55.69:8080/chainDemo/playground/
```

或者

```html
http://127.0.0.1:8080/chainDemo/playground/
```

此时我们再使用ApiPost发送POST情况，结果示例如下：

<div align=center>
    <img src=./imgs_markdown/2025-02-27-09-56-25.png
    width=100%></br><center></center>
</div>

---

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：为什么Windows和WSL2的IP是一样的？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：这说明我们的WSL2是以桥接（Bridge）或Hyper-V网络模式运行的，而不是默认的NAT（Network Address Translation）模式，因此WSL2和Windows共享了相同的IP地址。

通常，WSL2默认使用NAT模式，这意味着：
- Windows和WSL2应该在不同的子网，比如：
  - Windows: `192.168.1.100`
  - WSL2: `172.20.64.1`

如果它们有相同的IP，说明WSL2可能是桥接模式（Bridge Mode）或Hyper-V模式，让两个系统共享相同的IP。


我们可以手动检查WSL2的网络模式：

在 WSL2 终端运行：

```sh
ip route
```

```
default via 10.72.55.1 dev eth0 proto kernel metric 25 
10.72.55.0/24 dev eth0 proto kernel scope link metric 281 
10.72.55.1 dev eth0 proto kernel scope link metric 25
```

如果我们看到 `10.72.55.1` 作为网关，那说明 **WSL2 和 Windows 共享 IP**。如果我们看到 `172.x.x.x` 这样的 IP，说明 **WSL2 是 NAT 模式**。

| **场景**                              | **解决方案**                                           |
| ------------------------------------- | ------------------------------------------------------ |
| Windows 和 WSL2 共享 IP（桥接模式）   | 直接访问 `http://127.0.0.1:8080/chainDemo/playground/` |
| Windows 和 WSL2 有不同 IP（NAT 模式） | 访问 `http://<WSL2的IP>:8080/chainDemo/playground/`    |
| 如果 NAT 模式不可用                   | 修改 `.wslconfig` 并重启 WSL2                          |

---

🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：使用NAT模式好还是桥接模式好？

🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在 WSL2 中，**NAT（Network Address Translation）模式**和**桥接（Bridge）模式**各有优缺点，适用于不同的使用场景。让我们详细比较它们。  

<font color='blue'><b>【🔍 NAT 模式 vs. 桥接模式】</b></font>

| **特性**     | **NAT 模式（默认）**                            | **桥接模式**                                |
| ------------ | ----------------------------------------------- | ------------------------------------------- |
| IP 地址      | WSL2 有独立的 `172.x.x.x` 或 `192.168.x.x` IP   | WSL2 共享 Windows 的 IP（如 `10.72.55.69`） |
| 访问 Windows | Windows 可以直接访问 WSL2，但需要手动端口转发   | Windows 直接访问 WSL2，无需端口转发         |
| 访问外部网络 | WSL2 可以访问互联网，但可能需要手动配置端口转发 | WSL2 直接访问互联网，与 Windows 无差别      |
| 端口映射     | Windows 访问 WSL2 需要 `netsh portproxy`        | 端口自动映射，无需手动配置                  |
| 局域网访问   | WSL2 默认无法被局域网访问，需要额外配置         | 局域网设备可以直接访问 WSL2                 |
| 安全性       | 更安全，WSL2 仅能通过 Windows 访问              | 可能增加安全风险，WSL2 公开在局域网         |

<font color='blue'><b>【🚀 什么时候选择 NAT 模式（默认）？】</b></font>

1. **一般本地开发**（如 Python、FastAPI、Docker 开发）  
   - 我们只在本机（Windows）访问 WSL2 运行的服务，比如 `http://localhost:8080`。
   - 我们可以使用 `netsh interface portproxy` 将 Windows 端口映射到 WSL2 端口。
   - 这种模式更安全，WSL2 不会直接暴露在局域网。

2. **需要更高的安全性**  
   - NAT 模式下，WSL2 **不会直接暴露在局域网**，只有 Windows 可以访问它。
   - 如果我们不希望 WSL2 的服务器被其他设备访问（如家用 WiFi 其他设备），NAT 模式是更好的选择。

3. **默认 WSL2 使用 NAT 模式，兼容性最好**  
   - Windows 11 以及新版 Windows 10 默认使用 NAT 模式，所有官方文档和大部分教程都基于 NAT 模式。

<font color='green'><b>如何让 Windows 访问 WSL2（NAT 模式下）？</b></font>  

手动端口映射：

```powershell
# 在PowerShell（⚠️管理员模式）
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=<WSL2_IP>
```

然后在 Windows 访问：

```
http://127.0.0.1:8080
```

<font color='blue'><b>【🌍 什么时候选择桥接模式？】</b></font>

1. **需要在局域网（LAN）访问 WSL2 服务**  
   - 例如，我们的 WSL2 运行了一个 Web 服务器，我们希望同一 WiFi 下的手机、其他电脑能直接访问它。
   - NAT 模式下，WSL2 **默认不会被局域网访问**，但桥接模式可以。

2. **希望 WSL2 和 Windows 共享网络环境**  
   - 我们希望 WSL2 **使用 Windows 的 IP 地址**，这样所有网络服务（如 VPN、代理）都能直接在 WSL2 里生效。
   - 例如，我们的 Windows 连接了 VPN，我们希望 WSL2 也能直接走 VPN 访问网络。

3. **不想手动配置端口转发**  
   - 在 NAT 模式下，Windows 访问 WSL2 需要 `netsh portproxy` 进行端口转发。
   - 在桥接模式下，Windows 直接可以访问 WSL2，无需额外配置端口转发。

<font color='green'><b>如何开启桥接模式？</b></font>

1. **创建 `.wslconfig` 文件**
   在 Windows 运行：

   ```powershell
   # 在PowerShell（⚠️管理员模式）
   notepad C:\Users\你的用户名\.wslconfig
   ```

   添加以下内容：

   ```
   [wsl2]
   networkingMode=bridged
   ```

2. **关闭并重启 WSL2**
   ```powershell
   wsl --shutdown
   ```

   然后重新打开 WSL2，运行：

   ```sh
   ip addr show eth0
   ```

   我们应该会看到 WSL2 现在**使用和 Windows 相同的 IP 地址**。


<font color='blue'><b>【🎯 结论：应该选 NAT 还是桥接？】</b></font>

| 适用情况                            | 推荐模式       |
| ----------------------------------- | -------------- |
| **本机开发（Windows 访问 WSL2）**   | **NAT 模式** ✅ |
| **更高的安全性（防止局域网访问）**  | **NAT 模式** ✅ |
| **WSL2 需要局域网访问**             | **桥接模式** ✅ |
| **希望 WSL2 直接共享 Windows 网络** | **桥接模式** ✅ |

- **如果我们只是本地开发，NAT 模式是最好的选择**，因为它更安全，不用担心 WSL2 直接暴露在局域网里。  
- **如果我们希望局域网设备（如手机、其他电脑）访问 WSL2，桥接模式更方便**，但需要注意安全性。  

<font color='blue'><b>【🚀 最终建议】</b></font>

- **一般开发使用 NAT 模式**，安全性更高，兼容性最好。  
- **如果 WSL2 需要在局域网可访问，才使用桥接模式**。  
- **如果我们不确定，保持默认的 NAT 模式，并使用 `netsh portproxy` 让 Windows 访问 WSL2**。  

## 4.3. 案例二：聊天机器人

这个聊天机器人能够进行对话并记住之前的互动。这个聊天机器人有以下特性：

- **Chat History**：它允许聊天机器人“记住”过去的互动，并在回应后续问题时考虑它们。
- **流试输出**：它支持流式输出，这意味着它可以在每次输出一个字符时立即返回，而不是等待整个输出完成。

### 4.3.1. 安装langchain_community

```bash
pip install langchain_community
```

```
Installing collected packages: python-dotenv, mypy-extensions, marshmallow, httpx-sse, typing-inspect, pydantic-settings, dataclasses-json, langchain_community
Successfully installed dataclasses-json-0.6.7 httpx-sse-0.4.0 langchain_community-0.3.18 marshmallow-3.26.1 mypy-extensions-1.0.0 pydantic-settings-2.8.0 python-dotenv-1.0.1 typing-inspect-0.9.0
```

### 4.3.2. 代码

我们可以在LangSmith中创建一个项目：

```bash
export LANGCHAIN_PROJECT="LangChain入门"
```

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

def get_session_history(session_id: str) -> str:
    """
    根据会话ID获取会话历史记录。

    如果给定的会话ID在存储中不存在，那么会为该会话ID创建一个新的ChatMessageHistory实例，并存储以备后用。
    这个函数确保了每个会话ID都有一个对应的会话历史记录对象，无论是新创建的还是之前就存在的。

    参数:
        session_id (str): 会话的唯一标识符。

    返回:
        str: 对应于给定会话ID的会话历史记录。
    """
    # 检查给定的会话ID是否已存在于存储中
    if session_id not in chat_history_store:
        # 如果不存在，为该会话ID创建一个新的ChatMessageHistory实例
        chat_history_store[session_id] = ChatMessageHistory()
    # 返回与会话ID关联的会话历史记录对象
    return chat_history_store[session_id]

if __name__ == '__main__':
    # 创建LLM实例对象
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 定义提示词模板
    prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            ('system', '你是一个乐于助人的助手。用{language}尽你所能回答所有问题。'),
            MessagesPlaceholder(variable_name='chat_history')
        ],
        template_format='f-string'  # 如何解析{}中的内容，默认是f-string
    )

# 创建链
chain = prompt_template | model

# 保存聊天的历史记录（所有用户的聊天记录都会保存）
# key: sessionId
# value: 聊天记录
chat_history_store: dict = {}

# 创建一个RunnableWithMessageHistory实例，用于处理聊天记录
runner = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,  # 不要加括号
    input_messages_key='chat_history',  # 输入消息的键: 每次聊天时发送msg的key
)

# 配置
config = {
    'configurable': {'session_id': 'chat_example_1'}  # 给当前聊天会话设置一个session_id
}

# 第一轮聊天 -> 没有聊天记录
response = runner.invoke(
    input={
        'chat_history': [
            HumanMessage(
                content='你好，我是Le0v1n',  # 发给LLM的第一个聊天记录
            )
        ],
        'language': '中文'  # 因为之前的Template中定义了要用什么语言进行回答，所以在第一次聊天的时候我们也需要定义这个Language
    },
    config=config
)

print(f"[第一轮] {response.content = }")

# 第二轮聊天 -> 没有聊天记录
response = runner.invoke(
    input={
        'chat_history': [
            HumanMessage(
                content='请问，我的名字是什么？',  # 这个问题就可以看出来LLM会不会记住我们的名字
            )
        ],
        'language': '中文'  # 因为之前的Template中定义了要用什么语言进行回答，所以在第一次聊天的时候我们也需要定义这个Language
    },
    config=config
)

print(f"[第二轮] {response.content = }")

# 第三轮对话 -> 返回的数据是流式的
print('-' * 25 + ' [第三轮] ' + '-' * 25)
for resp in runner.stream( # 因为我们要使用流式输出，所以这里不再是invoke而是stream
    input={
        'chat_history': [
            HumanMessage(content='请给我介绍一下LangChain')
        ],
        'language': '中文'
    },
    config=config
):  # 每一次response都是一个token
    print(f"{resp.content}", end='')
print(f"")
```

```
[第一轮] response.content = '你好，Le0v1n！很高兴见到你。有什么我可以帮助你的吗？'
[第二轮] response.content = '你的名字是Le0v1n。有什么关于名字或其他方面的问题吗？'
------------------------- [第三轮] -------------------------
LangChain 是一个用于构建语言模型应用的框架，它旨在简化与自然语言处理（NLP）相关的开发过程。LangChain 的核心理念是将语言模型和相关的组件（如数据、工具、API等）结合起来，以便更好地处理复杂任务。

以下是 LangChain 的一些主要功能和特点：

1. **组件化设计**：LangChain 采用模块化的设计，可以将不同的组件（如链条、工具、数据源等）组合在一起，以满足特定的需求。

2. **链条（Chains）**：可以通过定义一个或多个链条来实现复杂的工作流。这些链条可以分为多个步骤，如数据处理、生成响应等。

3. **环境支持**：LangChain 支持多种环境，包括与外部 API 的交互、数据库访问等，这使得与其他服务集成变得更加简单。

4. **适用于多种任务**：LangChain 可以用于问答系统、文本生成、对话生成以及信息提取等多种 NLP 应用场景。

5. **接入多种语言模型**：用户可以选择不同的语言模型进行调用，支持的模型包括 OpenAI 的 GPT 系列等。

6. **文档和示例**：LangChain 提供了丰富的文档和示例，使得开发者可以更快上手和理解其用法。

LangChain 是一个非常有用的工具，特别适合需要构建基于语言模型的应用程序的开发者。如果你对某个具体功能感兴趣，或者有其他相关问题，欢迎随时问我！
```

## 4.4. 案例三：LangChian构建向量数据库和检索器

支持从向量数据库和其他来源检索数据，以便与LLM（大型语言模型）工作流程集成。它们对于应用程序来说非常重要，这些应用程序需要获取数据以作为模型推理的一部分进行推理，就像检索增强生成（RAG）的情况一样。

💡简单理解chroma的主要作用是为我们存储和检索向量数据库的，langchain-chroma可以简单理解为就是专门为langchain设计的向量数据库。简易流程图如下：

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px;
    
    A(文档):::process --> B(文档加载):::process
    B --> C(文本分割):::process
    C --> D(文本向量化):::process
    D --> E(Chroma 向量存储):::process
    F(用户问题):::process --> G(问题向量化):::process
    G --> H(Chroma 向量检索):::process
    E --> H
    H --> I(答案生成):::process
    I --> J(返回答案给用户):::process
```

**流程图说明**：

1. **文档处理阶段**：
    - 从 **文档** 开始，首先进行 **文档加载**，将文档读取到系统中。
    - 接着进行 **文本分割**，把文档拆分成更小的文本块，方便后续处理。
    - 对分割后的文本进行 **文本向量化**，将文本转换为向量表示。
    - 最后将这些向量存储到 **Chroma 向量存储** 中。

2. **用户交互阶段**：
    - 用户提出 **用户问题**，对问题进行 **问题向量化**，将问题转换为向量。
    - 在 **Chroma 向量检索** 中，根据问题向量从 Chroma 向量存储中查找相似的向量。
    - 检索到相关信息后，进行 **答案生成**，结合检索结果生成最终的答案。
    - 最后将 **答案返回给用户**。

> 🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：什么是chroma？
>
> 🥳 𝑨𝒏𝒔𝒘𝒆𝒓：LangChain和Chroma是两个在AI领域互补的工具，通常结合使用构建智能应用。**Chroma**是一个开源的向量数据库，核心功能：
>
> - 存储和检索嵌入向量（如文本/图像的特征向量）
> - 支持高效的相似性搜索（ANN近似最近邻算法）
> - 内置轻量级版本，支持云部署和分布式扩展
> - 与主流AI框架（Hugging Face、OpenAI）无缝集成
>
> **结合使用场景**  
> 在LangChain构建的问答系统中，Chroma常作为向量数据库存储文档/对话的嵌入向量，实现：
>
> - 上下文检索增强生成（RAG）
> - 个性化推荐
> - 对话历史管理
> - 知识库构建与查询
>
> 两者组合可快速实现从数据摄入、特征提取、智能处理到输出的完整AI应用链路。

### 4.4.1. 安装langchain-chroma

```bash
pip install langchain-chroma
```

### 4.4.2. 代码

#### 4.4.2.1. 构建向量空间并计算相似度

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 准备测试数据，假设我们提供的文档数据如下：
my_documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",  # 文档的核心文本内容
        metadata={"source": "哺乳动物宠物文档"},  # 元数据
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

if __name__ == '__main__':
    # 实例化向量数据库（向量空间）
    vector_store = Chroma.from_documents(
        documents=my_documents,
        embedding=OpenAIEmbeddings(base_url="https://www.dmxapi.com/v1")  # 使用什么应该的Embedding方式
    )

    # [示例1] 查询query与向量空间的相似度分数（⚠️分数越低，相似度越高）
    similar_scores = vector_store.similarity_search_with_score(
        query='暹罗猫'
    )
    for i, score in enumerate(similar_scores):
        print(f"[{i}] {score[0]} --> {score[1]}")

    # [示例2] 根据向量空间创建一个检索器并批量计算输入与向量空间的相似度
    retriever = RunnableLambda(
        func=vector_store.similarity_search
    ).bind(k=1)  # k=1表示选取相似度最高的那个

    # 批量检索多个query与向量空间的相似度
    queries: list = [
            '暹罗猫', 
            '小鲨鱼',
            'Le0v1n',
            '人类'
        ]
    similarities: list = retriever.batch(
        inputs=queries
    )

    for i, (query, similarity) in enumerate(zip(queries, similarities)):
        print(f"[{i}] {query} --> {similarity}")
```

```
[0] page_content='猫是独立的宠物，通常喜欢自己的空间。' metadata={'source': '哺乳动物宠物文档'} --> 0.34998974204063416
[1] page_content='兔子是社交动物，需要足够的空间跳跃。' metadata={'source': '哺乳动物宠物文档'} --> 0.3990579843521118
[2] page_content='狗是伟大的伴侣，以其忠诚和友好而闻名。' metadata={'source': '哺乳动物宠物文档'} --> 0.41659438610076904
[3] page_content='金鱼是初学者的流行宠物，需要相对简单的护理。' metadata={'source': '鱼类宠物文档'} --> 0.4340311288833618

[0] 暹罗猫 --> [Document(id='9e95e892-3362-4239-87d3-2bb08bab72da', metadata={'source': '哺乳动物宠物文档'}, page_content='猫是独立的宠物，通常喜欢自己的空间。')]
[1] 小鲨鱼 --> [Document(id='b21bebf7-79ca-4262-9c29-ae1bf883c5e7', metadata={'source': '鱼类宠物文档'}, page_content='金鱼是初学者的流行宠物，需要相对简单的护理。')]
[2] Le0v1n --> [Document(id='0b2ef424-a9b3-40fc-a112-f443cddf88d1', metadata={'source': '哺乳动物宠物文档'}, page_content='兔子是社交动物，需要足够的空间跳跃。')]
[3] 人类 --> [Document(id='a773d28c-59e2-4b9d-95d6-5843dd2e8918', metadata={'source': '鸟类宠物文档'}, page_content='鹦鹉是聪明的鸟类，能够模仿人类的语言。')]
```

根据示例1的结果我们可以看到，当我们计算相似度时，分数越低表示越相似。导致这样反直觉的原因是该方法虽然使用的是余弦相似度，<font color='red'><b>但Chroma进行了特殊的处理，导致分数越低越相似</b></font>，可以查看文档：[https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.similarity_search_with_score](https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.similarity_search_with_score)。

我们介绍一下常见的相似度计算方式：

<font color='blue'><b>常见相似度度量及分数含义</b></font>

| 相似度度量方法 | 原理                                                                                   | 取值范围 | 分数与相似度关系                                                                      | 优点                                                                     | 缺点                                                                 | 适用场景                               |
| -------------- | -------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- | -------------------------------------- |
| 余弦相似度     | 通过计算两个向量之间夹角的余弦值来衡量相似度，不考虑向量的模长，仅关注方向             | [-1, 1]  | 分数越接近 1 ，表示两个向量方向越相似，文本内容越相似；分数越接近 -1 ，表示方向越相反 | 不受向量长度影响，适用于处理文本特征向量，能够有效衡量文本语义上的相似性 | 没有考虑向量的大小，对于向量长度有重要意义的场景不太适用             | 文本分类、信息检索、语义分析等         |
| 欧几里得距离   | 计算两个向量在空间中的实际直线距离                                                     | [0, +∞)  | 距离越小，两个向量越接近，文本越相似                                                  | 直观易懂，符合人们对空间距离的认知，在很多实际问题中有明确的物理意义     | 受向量尺度影响较大，如果向量各维度的尺度差异较大，需要进行归一化处理 | 聚类分析、图像识别、地理信息系统等     |
| 曼哈顿距离     | 计算两个向量在各个维度上差值的绝对值之和                                               | [0, +∞)  | 距离越小，两个向量越接近，文本越相似                                                  | 计算简单，对异常值相对不敏感                                             | 没有考虑向量的方向信息，只考虑了各个维度上的绝对差值                 | 网格地图路径规划、特征匹配等           |
| 杰卡德相似度   | 用于比较有限样本集之间的相似度，通过计算两个集合的交集元素个数与并集元素个数的比值得到 | [0, 1]   | 分数越接近 1 ，表示两个集合的相似性越高；分数越接近 0 ，表示相似性越低                | 适用于处理集合数据，如文本中的词集、图像中的特征集等                     | 只考虑集合元素的存在与否，不考虑元素的重复次数                       | 文本抄袭检测、图像匹配、社交网络分析等 |

接下来我们简单介绍一下各种相似度：

<font color='red'><b>【余弦相似度】</b></font>

- **原理**：余弦相似度通过计算两个向量之间夹角的余弦值来衡量它们的相似度。余弦值的范围是 `[-1, 1]`，值越接近 1 表示两个向量的方向越相似，也就意味着文本内容越相似。
- **在 `Chroma` 中的体现**：在 `Chroma` 里，如果使用余弦相似度作为度量方式，`similarity_search_with_score` 返回的分数通常就是余弦相似度值，此时越相似分数越高。

<font color='red'><b>【欧几里得距离】</b></font>

- **原理**：欧几里得距离计算的是两个向量在空间中的实际距离。距离越小，说明两个向量越接近，也就表示文本越相似。
- **在 `Chroma` 中的体现**：如果 `Chroma` 使用欧几里得距离作为相似度度量，`similarity_search_with_score` 返回的分数可能就是欧几里得距离值，这种情况下越相似分数越低。不过，`Chroma` 默认一般使用余弦相似度，若要使用欧几里得距离，可能需要额外的配置。

#### 4.4.2.2. 将检索器和LLM结合在一起

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

# 准备测试数据，假设我们提供的文档数据如下：
my_documents = [
    Document(
        page_content="狗是伟大的伴侣，以其忠诚和友好而闻名。",  # 文档的核心文本内容
        metadata={"source": "哺乳动物宠物文档"},  # 元数据
    ),
    Document(
        page_content="猫是独立的宠物，通常喜欢自己的空间。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类宠物文档"},
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类宠物文档"},
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"},
    ),
]

if __name__ == '__main__':
    # 实例化向量数据库（向量空间）
    vector_store = Chroma.from_documents(
        documents=my_documents,
        embedding=OpenAIEmbeddings(base_url="https://www.dmxapi.com/v1")  # 使用什么应该的Embedding方式
    )

    # 根据向量空间创建一个检索器
    retriever = RunnableLambda(
        func=vector_store.similarity_search
    ).bind(k=1)  # k=1表示选取相似度最高的那个

    # 定义Prompt Template
    message = """使用提供的上下文（向量空间/向量数据库）来回答这个问题：{question}。
    上下文：{context}"""

    prompt_template = ChatPromptTemplate.from_messages(
        messages=[  # 需要传入Sequence，即tuple、list、str
            ('human', message)
        ]  
    )

    # 构建LLM模型
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 构建chain（因为我们在prompt_template定义了message，所以我们应该在构建chain的时候应该是先将question和context传入）
    chain = {
        # ⚠️这里的RunnablePassThrough应该加上()
        'question': RunnablePassthrough(),  # RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model
        'context': retriever,
    } | prompt_template | model

    resp = chain.invoke(
        input='请介绍一下猫'
    )
    print(f"{resp.content = }")
```

```
resp.content = '猫是一种独立的宠物，通常喜欢拥有自己的空间。它们以独特的性格和习性而受到人们的喜爱。猫的行为通常比较自主，它们喜欢探索周围的环境，同时也能独立处理自己的事务。'
```

### 4.4.3. 语法点

#### 4.4.3.1. langchain_core.documents.Document

`langchain_core.documents.Document` 是 LangChain 框架中用于表示文档数据的核心类，在处理非结构化文本数据时扮演着重要角色。

**【类定义及用途】**

`Document` 类用于封装文本内容及其相关元数据。在 LangChain 的各类应用场景中，如文档问答、文本摘要、信息检索等，都需要将原始的文本数据转换为 `Document` 对象，以便后续处理。

**【主要属性】**该类主要包含两个属性：

- **`page_content`**：这是文档的核心文本内容，通常是一段连续的字符串。例如，当我们处理一篇新闻文章时，`page_content` 就是文章的具体文字内容。
- **`metadata`**：这是一个字典类型的数据，用于存储与文档相关的额外信息。这些信息可以是文档的来源、创建时间、作者、文档类型等。例如，对于一篇从网页上抓取的文章，`metadata` 中可以包含网页的 URL、抓取时间等信息。

**【代码示例】**

以下是一个简单的代码示例，展示如何创建 `Document` 对象：

```python
from langchain_core.documents import Document

# 创建一个 Document 对象
doc = Document(
    page_content="这是一段示例文本，用于演示 Document 类的使用。",
    metadata={"source": "示例文档", "date": "2024-01-01"}
)

# 打印文档内容和元数据
print("文档内容:", doc.page_content)
print("文档元数据:", doc.metadata)
```

```
文档内容: 这是一段示例文本，用于演示 Document 类的使用。
文档元数据: {'source': '示例文档', 'date': '2024-01-01'}
```

**【实际应用】**

在 LangChain 中，`Document` 对象被广泛应用于各种组件和工具中。例如：

- 在<font color='red'><b>文档加载器</b></font>中，会将不同格式的文档（如 `PDF`、`Word`、`CSV` 等）读取并转换为 `Document` 对象列表。
- 在<font color='red'><b>向量数据库</b></font>中，会将 `Document` 对象中的文本内容进行向量化处理，并将向量和元数据一起存储；在检索器中，会根据用户的查询从存储的 `Document` 对象中检索相关文档。

通过使用 `langchain_core.documents.Document` 类，开发者可以更方便地管理和处理文本数据，提高开发效率。 

#### 4.4.3.2. langchain_core.runnables.RunnablePassthrough

`langchain_core.runnables.RunnablePassthrough` 是 LangChain 框架中一个较为实用的组件，主要用于在链式操作中直接传递输入，不做任何修改，下面为我们详细介绍其作用：

**【基本作用】**

`RunnablePassthrough` 可作为一种<font color='red'><b>占位符</b></font>或连接器，允许我们在构建复杂的 LangChain 可运行对象链时，将输入原样传递给下一个步骤，而无需对输入进行额外的处理。这在需要将同一输入同时传递给多个不同的操作或者按顺序传递给不同组件时非常有用。

**【代码示例及说明】**

以下是一个简单的代码示例，展示了 `RunnablePassthrough` 的使用：

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# 定义一个聊天模型
model = ChatOpenAI()

# 定义一个提示模板
prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}")
])

# 创建一个 RunnablePassthrough 对象
passthrough = RunnablePassthrough()

# 将 RunnablePassthrough 与提示模板和模型组合成一个链式操作
chain = passthrough | prompt | model

# 输入数据
input_data = "给我讲一个简短的笑话。"

# 运行链式操作
output = chain.invoke({"input": input_data})

print(output)
```

在上述示例中：
1. `RunnablePassthrough` 对象 `passthrough` 被创建，它会直接将输入传递给下一个步骤。
2. `passthrough` 与 `prompt` 和 `model` 组合成一个链式操作 `chain`。
3. 当调用 `chain.invoke` 时，输入数据 `input_data` 首先经过 `passthrough`，它不做任何修改，直接将数据传递给 `prompt` 进行模板填充，然后将填充后的提示传递给 `model` 进行推理，最终得到输出结果。

```mermaid
graph LR
    classDef process fill:#E5F6FF,stroke:#73A6FF,stroke-width:2px;
    classDef input fill:#FFF6CC,stroke:#FFBC52,stroke-width:2px;
    
    A(输入数据: 给我讲一个简短的笑话):::input --> B(RunnablePassthrough):::process
    B --> C(ChatPromptTemplate):::process
    C --> D(ChatOpenAI 模型):::process
    D --> E(输出结果):::input
```

**【实际应用场景】**

- **多分支处理**：当需要将同一输入同时传递给多个不同的处理流程时，可以使用 `RunnablePassthrough` 作为起点，将输入复制并传递给各个分支。
- **顺序处理流程**：在构建复杂的处理流程时，某些步骤可能不需要对输入进行修改，只需要将其传递给下一个步骤，这时就可以使用 `RunnablePassthrough` 来简化链式操作的构建。

通过使用 `RunnablePassthrough`，可以更灵活地构建 LangChain 应用程序的处理流程，提高代码的可读性和可维护性。

## 4.5. 案例四：Langchain构建Agent

语言模型本身无法执行动作，它们只能输出文本。Agent是使用大型语言模型（LLM）作为推理引擎来确定要执行的操作，以及这些操作的输入应该是什么。然后，这些操作的结果可以反馈到Agent中，Agent将决定是否需要更多的操作，或者是否可以结束。

我们要构建Agent需要使用Langchain的库——`langgraph`。

### 4.5.1. 安装langgraph

```bash
pip install langgraph
```

### 4.5.2. langgraph简介

LangGraph是LangChain生态中的重要扩展框架，专为构建复杂、有状态的智能体（Agent）应用而设计。它通过可视化工作流、灵活的控制逻辑和状态管理能力，帮助开发者高效创建多步骤、可定制的AI系统。

#### 4.5.2.1. 核心功能与特点

1. **可视化流程设计**：提供图形化界面，直观展示任务节点（如LLM调用、工具执行）和数据流，支持循环、条件分支等复杂逻辑，便于调试和优化多步骤工作流。

2. **精细控制与灵活性**：允许开发者自定义智能体的决策过程，支持条件判断、动态流程调整，甚至人机协作（如人工审核介入），适用于金融、医疗等高敏感场景。

3. **状态持久化与记忆**：自动保存每个步骤的状态，支持会话恢复、历史追溯，确保多轮交互的连贯性，适合需要上下文记忆的对话系统或任务自动化。

4. **与LangChain生态深度集成**：无缝对接LangChain的工具链和LangSmith的监控功能，同时支持独立运行，兼容Python和JavaScript，扩展性强。

#### 4.5.2.2. 典型应用场景

- **智能客服/聊天机器人**：管理多轮对话并保持上下文。
- **自动化任务系统**：串联检索、分析、决策等步骤。
- **复杂问题解决**：如法律咨询、数据分析等需要多工具协作的场景。

#### 4.5.2.3. 与其他工具的区别

- **对比LangFlow**：LangFlow侧重低代码快速原型，而LangGraph更强调对智能体行为的深度控制和状态管理。
- **对比LangSmith**：LangSmith专注于监控和调试，LangGraph则聚焦于构建阶段的流程设计与执行逻辑。

LangGraph通过开源框架和配套IDE（LangGraph Studio），为开发者提供了从设计到部署的全链条支持，是构建企业级AI智能体的理想选择。

### 4.5.3. 如果不使用Agent（直接使用LLM）查询天气

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

if __name__ == '__main__':
    # 构建LLM模型
    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    resp = model.invoke(
        input=[
            HumanMessage(
                content='西安今天的天气怎么样？'
            )
        ]
    )

    print(f"{resp.content = }")
```

```
resp.content = '对不起，我无法提供实时的天气信息。建议你查看天气预报网站或使用天气应用程序来获取西安今天的天气情况。如果你需要了解天气的常见趋势或季节特点，我很乐意帮助！'
```

我们可以看到，直接使用LLM是无法获取最新的信息的。那怎么解决这个问题？这就需要Agent了。

### 4.5.4. 使用绑定了工具的LLM查询天气

> 💡Tavily API网站：[https://app.tavily.com/](https://app.tavily.com/)

> 🚨在Langchain中，核心对象就是`langchain_core.runnables.Runnable`，只要是这个对象就可以调用了，如`.invoke()`

```python
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

if __name__ == '__main__':
    # 构建LLM模型
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 构建Tavily搜索引擎工具
    search_tool: Runnable = TavilySearchResults(
        max_results=2,  # 返回结果的最大数量
    )

    # 让LLM模型绑定工具
    model_with_tools: Runnable = model.bind_tools(
        tools=[  # tools参数要求是一个Sequence，所以我们这里使用了list
            search_tool
        ],
    )

    # 调用绑定了工具的LLM
    resp1 = model_with_tools.invoke(
        input=[
            HumanMessage(
                content='你是谁？'
            )
        ]
    )

    resp2 = model_with_tools.invoke(
        input=[
            HumanMessage(
                content='西安今天的天气怎么样？'
            )
        ]
    )

    # 分别打印LLM的结果和工具的结果
    print(f"[Response 1] [Model Result] [Content] {resp1.content}")
    print(f"[Response 1] [Tools Result] [Content] {resp1.tool_calls}")

    print(f"[Response 2] [Model Result] [Content] {resp2.content}")
    print(f"[Response 2] [Tools Result] [Content] {resp2.tool_calls}")
```

```
[Response 1] [Model Result] [Content] 我是一个人工智能助手，旨在提供信息和回答问题。如果你有什么疑问或需要帮助的地方，请随时告诉我！
[Response 1] [Tools Result] [Content] []

[Response 2] [Model Result] [Content] 
[Response 2] [Tools Result] [Content] [{'name': 'tavily_search_results_json', 'args': {'query': '西安 今日天气'}, 'id': 'call_w8K5WiVcro3rWfx6CXzcGYQ3', 'type': 'tool_call'}]
```

- 对于第一个问题，LLM是可以直接回答的，那么就不会调用工具，因此`[Tools Result] [Content] []`返回的是一个空的list。
- 对于第二个问题，LLM是无法直接回答的，因此LLM选择去调用工具。主要注意的是，这里返回的是一个回调函数，<font color='red'><b>只是表明LLM会调用工具，但实际上没有调用</b></font>。

### 4.5.5. 使用Agent查询天气

刚才我们看了绑定了工具的LLM去查询天气，发现LLM只是准备去调用工具，但实际上并没有真正的调用，为了让LLM可以去调用工具，这里就需要我们的Agent了。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import chat_agent_executor

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

if __name__ == '__main__':
    # 构建LLM模型
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 构建Tavily搜索引擎工具
    search_tool: Runnable = TavilySearchResults(
        max_results=2,  # 返回结果的最大数量
    )

    # 先把工具放到一起
    tools: list = [search_tool]

    # 创建一个Agent
    agent_executor: Runnable = chat_agent_executor.create_react_agent(
        model=model,
        tools=tools
    )

    # 示例1：调用Agent
    resp_1 = agent_executor.invoke(
        input={
            'messages': HumanMessage(
                content='你是谁？'
            )
        },
    )
    print(resp_1['messages'])  # 在调用的时候用的什么key，在取的时候我们就可以用这个key来取

    # 示例2：调用Agent
    resp_2 = agent_executor.invoke(
        input={
            'messages': HumanMessage(
                content='西安今天的天气怎么样？'
            )
        },
    )
    print(resp_2['messages'])  # 在调用的时候用的什么key，在取的时候我们就可以用这个key来取
```

```
[HumanMessage(content='你是谁？', additional_kwargs={}, response_metadata={}, id='8ce0f39d-690f-487c-9d2e-f813ba2334c5'), AIMessage(content='我是一个人工智能助手，旨在提供信息和回答问题。如果你有任何问题或需要帮助，请随时告诉我！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 82, 'total_tokens': 111, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'stop', 'logprobs': None}, id='run-1531caf6-724e-4477-b65f-3e2f7fb5a198-0', usage_metadata={'input_tokens': 82, 'output_tokens': 29, 'total_tokens': 111, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]

[HumanMessage(content='西安今天的天气怎么样？', additional_kwargs={}, response_metadata={}, id='0ef712b5-9523-441b-bb76-c0f4e4840ba3'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QZ1TFyOweOqX6oluuy4LFVvP', 'function': {'arguments': '{"query":"西安 今日天气"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 86, 'total_tokens': 108, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-30ddb8c6-a50f-4fea-b3a3-46abeaa31ec6-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': '西安 今日天气'}, 'id': 'call_QZ1TFyOweOqX6oluuy4LFVvP', 'type': 'tool_call'}], usage_metadata={'input_tokens': 86, 'output_tokens': 22, 'total_tokens': 108, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='[{"url": "https://tianqi.fliggy.com/610100/", "content": "果洛天气\\n河池天气\\n四平天气\\n赣州天气\\n呼和浩特天气\\n嘉兴天气\\n黔南天气\\n镇江天气\\n芜湖天气\\n石家庄天气\\n南通天气\\n乐山天气\\n池州天气\\n南充天气\\n长春天气\\n鞍山天气\\n盘锦天气\\n李民烧鸡老店天气\\n爱麦思休闲吧天气\\n八旗乌拉满族火锅天气\\n东山浴场天气\\n雪列车天气\\n乔一乔怪味餐厅(东升店)天气\\n粗茶淡饭·汉口公馆天气\\nGucci(成都天府机场店)天气\\n小公牛酒楼天气\\n重庆鼎味美蛙鱼头天气\\n韩式旋转小火锅(金利来店)天气\\n罗莎蛋糕(橙子街店)天气\\n聚香乐酒楼天气\\n先农坛天气\\n食在湘寨天气\\n港式烧腊天气\\n食悦麻辣香锅天气\\n\\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\\n关注飞猪旅行天气助手 [...] 西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\\n西安 24 小时天气预报\\n最近更新: 01:59\\n0:00\\n阴 [...] 2025年2月8日 星期六 农历正月十一最近更新: 01:59\\n1℃最低 0℃ - 最高 5℃\\n阴\\n风向：东北风\\n降水量：0.00mm\\n风速：10km/h\\n相对湿度：44%rh\\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\\n2025年2月8日 星期六 农历正月十一最近更新: 01:59\\n1℃最低 0℃ - 最高 5℃\\n阴\\n风向：东北风\\n降水量：0.00mm\\n风速：10km/h\\n相对湿度：44%rh"}, {"url": "https://tianqi.so.com/weather/101110101", "content": "全国天气网\\n首页\\n国内天气\\n空气质量\\n国际天气\\n景点天气\\n天气新闻\\n专业天气\\n收藏\\n西安[切换]当前时间：2025-02-09周日22:18\\n2\\n晴\\n空气 良西南风 1级\\n今天 (02-09)\\n多云转晴\\n0/7℃\\n持续无风向 微风\\n明天 (02-10)\\n多云转阴\\n0/9℃\\n持续无风向 微风\\n周二 (02-11)\\n小雨转阴\\n0/7℃\\n持续无风向 微风\\n周三 (02-12)\\n阴转多云\\n0/7℃\\n持续无风向 微风\\n周四 (02-13)\\n阴\\n0/8℃\\n持续无风向 微风\\n周五 (02-14)\\n阴\\n1/11℃\\n持续无风向 微风\\n周六 (02-15)\\n阴转多云\\n0/15℃\\n持续无风向 微风\\n周日 (02-16)\\n多云转晴\\n0/10℃\\n持续无风向 微风\\n周一 (02-17)\\n晴转多云\\n4/16℃\\n西南风 微风\\n周二 (02-18)\\n阴\\n6/16℃\\n南风 微风\\n周三 (02-19)\\n小雨转阴\\n3/8℃\\n西北风 微风\\n周四 (02-20)\\n阴转小雨\\n3/13℃\\n东北风 微风\\n周五 (02-21)\\n小雨转雨夹雪\\n1/6℃\\n东北风 微风\\n周六 (02-22)\\n雨夹雪转晴\\n0/3℃\\n东北风 微风"}]', name='tavily_search_results_json', id='ad57eec5-5ec0-4063-8f87-84ad074c278b', tool_call_id='call_QZ1TFyOweOqX6oluuy4LFVvP', artifact={'query': '西安 今日天气', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://tianqi.fliggy.com/610100/', 'title': '西安旅游指数/交通指数，24小时/未来天气预报查询 - 飞猪天气预报', 'content': '果洛天气\n河池天气\n四平天气\n赣州天气\n呼和浩特天气\n嘉兴天气\n黔南天气\n镇江天气\n芜湖天气\n石家庄天气\n南通天气\n乐山天气\n池州天气\n南充天气\n长春天气\n鞍山天气\n盘锦天气\n李民烧鸡老店天气\n爱麦思休闲吧天气\n八旗乌拉满族火锅天气\n东山浴场天气\n雪列车天气\n乔一乔怪味餐厅(东升店)天气\n粗茶淡饭·汉口公馆天气\nGucci(成都天府机场店)天气\n小公牛酒楼天气\n重庆鼎味美蛙鱼头天气\n韩式旋转小火锅(金利来店)天气\n罗莎蛋糕(橙子街店)天气\n聚香乐酒楼天气\n先农坛天气\n食在湘寨天气\n港式烧腊天气\n食悦麻辣香锅天气\n\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n关注飞猪旅行天气助手 [...] 西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n西安 24 小时天气预报\n最近更新: 01:59\n0:00\n阴 [...] 2025年2月8日 星期六 农历正月十一最近更新: 01:59\n1℃最低 0℃ - 最高 5℃\n阴\n风向：东北风\n降水量：0.00mm\n风速：10km/h\n相对湿度：44%rh\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n2025年2月8日 星期六 农历正月十一最近更新: 01:59\n1℃最低 0℃ - 最高 5℃\n阴\n风向：东北风\n降水量：0.00mm\n风速：10km/h\n相对湿度：44%rh', 'score': 0.8615960586956523, 'raw_content': None}, {'url': 'https://tianqi.so.com/weather/101110101', 'title': '【西安天气预报】西安天气预报7天,10天,15天_全国天气网', 'content': '全国天气网\n首页\n国内天气\n空气质量\n国际天气\n景点天气\n天气新闻\n专业天气\n收藏\n西安[切换]当前时间：2025-02-09周日22:18\n2\n晴\n空气 良西南风 1级\n今天 (02-09)\n多云转晴\n0/7℃\n持续无风向 微风\n明天 (02-10)\n多云转阴\n0/9℃\n持续无风向 微风\n周二 (02-11)\n小雨转阴\n0/7℃\n持续无风向 微风\n周三 (02-12)\n阴转多云\n0/7℃\n持续无风向 微风\n周四 (02-13)\n阴\n0/8℃\n持续无风向 微风\n周五 (02-14)\n阴\n1/11℃\n持续无风向 微风\n周六 (02-15)\n阴转多云\n0/15℃\n持续无风向 微风\n周日 (02-16)\n多云转晴\n0/10℃\n持续无风向 微风\n周一 (02-17)\n晴转多云\n4/16℃\n西南风 微风\n周二 (02-18)\n阴\n6/16℃\n南风 微风\n周三 (02-19)\n小雨转阴\n3/8℃\n西北风 微风\n周四 (02-20)\n阴转小雨\n3/13℃\n东北风 微风\n周五 (02-21)\n小雨转雨夹雪\n1/6℃\n东北风 微风\n周六 (02-22)\n雨夹雪转晴\n0/3℃\n东北风 微风', 'score': 0.8307216260869565, 'raw_content': None}], 'response_time': 1.45}), AIMessage(content='今天（2月9日），西安的天气预报如下：\n\n- **气温**：最低0℃，最高7℃\n- **天气状况**：多云转晴\n- **风向**：西南风，风速1级\n- **降水量**：无降水\n- **相对湿度**：当前空气质量良好\n- **生活指数**：\n  - 交通指数：良好（天气较好，路面干燥）\n  - 旅游指数：适宜（天气适合出行）\n\n整体来看，今天西安的天气非常适合外出活动。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 137, 'prompt_tokens': 1427, 'total_tokens': 1564, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'stop', 'logprobs': None}, id='run-68b3f2b8-6ce2-44d8-913f-99661ad16e4b-0', usage_metadata={'input_tokens': 1427, 'output_tokens': 137, 'total_tokens': 1564, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

#### 4.5.5.1. 第一个回答的分析

对于第一个回答，我们的提问是：`你是谁？`。

```python
[HumanMessage(content='你是谁？', additional_kwargs={}, response_metadata={}, id='8ce0f39d-690f-487c-9d2e-f813ba2334c5'), AIMessage(content='我是一个人工智能助手，旨在提供信息和回答问题。如果你有任何问题或需要帮助，请随时告诉我！', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 29, 'prompt_tokens': 82, 'total_tokens': 111, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'stop', 'logprobs': None}, id='run-1531caf6-724e-4477-b65f-3e2f7fb5a198-0', usage_metadata={'input_tokens': 82, 'output_tokens': 29, 'total_tokens': 111, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

对于这个问题，LLM本身就可以回答，所以：

- `AIMessage(content='我是一个人工智能助手，旨在提供信息和回答问题。如果你有任何问题或需要帮助，请随时告诉我！',`：`AIMessage`这个里面是有回答的，并且`AIMessage`也表明是LLM直接回答的。

#### 4.5.5.2. 第二个回答的分析

对于第一个回答，我们的提问是：`西安今天的天气怎么样？`。

```python
[HumanMessage(content='西安今天的天气怎么样？', additional_kwargs={}, response_metadata={}, id='0ef712b5-9523-441b-bb76-c0f4e4840ba3'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QZ1TFyOweOqX6oluuy4LFVvP', 'function': {'arguments': '{"query":"西安 今日天气"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 86, 'total_tokens': 108, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-30ddb8c6-a50f-4fea-b3a3-46abeaa31ec6-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': '西安 今日天气'}, 'id': 'call_QZ1TFyOweOqX6oluuy4LFVvP', 'type': 'tool_call'}], usage_metadata={'input_tokens': 86, 'output_tokens': 22, 'total_tokens': 108, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content='[{"url": "https://tianqi.fliggy.com/610100/", "content": "果洛天气\\n河池天气\\n..................\\n0/3℃\\n东北风 微风"}]', name='tavily_search_results_json', id='ad57eec5-5ec0-4063-8f87-84ad074c278b', tool_call_id='call_QZ1TFyOweOqX6oluuy4LFVvP', artifact={'query': '西安 今日天气', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'url': 'https://tianqi.fliggy.com/610100/', 'title': '西安旅游指数/交通指数，24小时/未来天气预报查询 - ................风速：10km/h\n相对湿度：44%rh', 'score': 0.8615960586956523, 'raw_content': None}, {'url': 'https://tianqi.so.com/weather/101110101', 'title': '【西安天气预报】西安天气预报7天,10天,15天_全国天气网', 'content': '全国天气网\n首页\n国内天气\n空气质量\n国际天气\n景点天气\n天气新闻\n专业天气\n收藏\n西安[切换]当前时间：2025-02-09周日22:18\n2\n晴\n空气 良西南风 1级\n今天 (02-09)\n多云转晴\n0/7℃\n持续无风向 微风\n.............雨夹雪转晴\n0/3℃\n东北风 微风', 'score': 0.8307216260869565, 'raw_content': None}], 'response_time': 1.45}), AIMessage(content='今天（2月9日），西安的天气预报如下：\n\n- **气温**：最低0℃，最高7℃\n- **天气状况**：多云转晴\n- **风向**：西南风，风速1级\n- **降水量**：无降水\n- **相对湿度**：当前空气质量良好\n- **生活指数**：\n  - 交通指数：良好（天气较好，路面干燥）\n  - 旅游指数：适宜（天气适合出行）\n\n整体来看，今天西安的天气非常适合外出活动。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 137, 'prompt_tokens': 1427, 'total_tokens': 1564, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_b705f0c291', 'finish_reason': 'stop', 'logprobs': None}, id='run-68b3f2b8-6ce2-44d8-913f-99661ad16e4b-0', usage_metadata={'input_tokens': 1427, 'output_tokens': 137, 'total_tokens': 1564, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
```

对于这个问题，LLM本身是无法回答的，然而我们使用的Agent，所以这里的回答也挺有意思的：

- `AIMessage(content='', `：`AIMessage`里面的`content`是空的，说明LLM没有直接回答
- `ToolMessage(content='[{"url": "https://tianqi.fliggy.com/610100/", "content": "果洛天气\\n河池天气\\n.................`：我们看到了`ToolMessage`，里面是有内容的，都是工具搜索的信息，说明这里是工具在起作用。我们分析一下这个工具返回的内容：
  - 先给出了url
  - 每个url后面也会有搜索到的内容

此时我们发现一个，虽然Agent可以调用工具，但好像只是调用了工具，没有输出我们想要的，比如直接输出今天的天气情况，那么我们应该怎么做呢？



### 4.5.6. Tavily简介

Tavily是一款专为人工智能代理（如大型语言模型）和检索增强生成（RAG）应用设计的搜索引擎，旨在提供高效、准确的实时信息支持。以下是其核心特点与价值：

#### 4.5.6.1. 核心功能与特点

1. **专为LLM优化**  
   - 直接整合搜索、抓取、过滤和内容提取全流程，通过单次API调用返回结构化结果，减少开发者负担。
   - 提供“基本”和“高级”搜索模式，高级模式耗时稍长但内容相关性更高，适合复杂任务。

2. **智能结果处理**  
   - 自动生成适用于LLM的上下文摘要，支持自定义最大令牌限制，优化RAG应用中的上下文输入。
   - 提供问答搜索（QNA Search）功能，直接返回包含来源的简明答案，降低模型幻觉风险。

3. **灵活集成与扩展性**  
   - 支持Python和JavaScript SDK，提供异步客户端（AsyncTavilyClient）以应对高并发场景。
   - 与LangChain、LlamaIndex等框架深度集成，兼容无代码平台（如集简云），适配不同开发需求。

#### 4.5.6.2. 核心优势

- **高效可靠**：通过AI算法筛选可信来源，减少低质量内容干扰，提升信息准确性。
- **实时性**：快速获取最新数据，适用于新闻聚合、金融分析等对时效性敏感的场景。
- **开发者友好**：提供丰富的参数配置（如时间范围、领域过滤）和详细文档，降低集成难度。

#### 4.5.6.3. 典型应用场景

- **智能客服/助手**：结合LLM生成基于实时数据的精准回答。
- **内容生成**：为RAG应用提供事实性上下文，增强生成内容的可信度。
- **研究与数据分析**：快速聚合多源信息，辅助决策支持或学术研究。

#### 4.5.6.4. 与其他工具的对比

- **对比传统搜索引擎API**（如SerpAPI）：Tavily专注于LLM需求，直接输出可用于RAG的精炼内容，而非简单URL列表。
- **对比通用搜索工具**：通过领域定制（如金融、代码库）和动态过滤，更贴合垂直场景需求。

Tavily通过免费API信用额度（每月1000次）和灵活的付费计划，成为开发者构建企业级AI应用的重要工具，尤其在需要实时信息支撑的场景中表现突出。

#### 4.5.6.5. 代码示例

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# 构建Tavily搜索引擎工具
search = TavilySearchResults(
    max_results=2,  # 返回结果的最大数量
)

# 尝试使用Tavily工具
resp = search.invoke('西安今天的天气怎么样？')
print(f"{resp = }")
```

```python
resp = [
    {
        'url': 'https://tianqi.so.com/weather/101110101', 
        'content': '全国天气网\n首页\n国内天气\n空气质量\n国际天气\n景点天气\n天气新闻\n专业天气\n收藏\n西安[切换]当前时间：2025-02-09周日22:18\n2\n晴\n空气 良西南风 1级\n今天 (02-09)\n多云转晴\n0/7℃\n持续无风向 微风\n明天 (02-10)\n多云转阴\n0/9℃\n持续无风向 微风\n周二 (02-11)\n小雨转阴\n0/7℃\n持续无风向 微风\n周三 (02-12)\n阴转多云\n0/7℃\n持续无风向 微风\n周四 (02-13)\n阴\n0/8℃\n持续无风向 微风\n周五 (02-14)\n阴\n1/11℃\n持续无风向 微风\n周六 (02-15)\n阴转多云\n0/15℃\n持续无风向 微风\n周日 (02-16)\n多云转晴\n0/10℃\n持续无风向 微风\n周一 (02-17)\n晴转多云\n4/16℃\n西南风 微风\n周二 (02-18)\n阴\n6/16℃\n南风 微风\n周三 (02-19)\n小雨转阴\n3/8℃\n西北风 微风\n周四 (02-20)\n阴转小雨\n3/13℃\n东北风 微风\n周五 (02-21)\n小雨转雨夹雪\n1/6℃\n东北风 微风\n周六 (02-22)\n雨夹雪转晴\n0/3℃\n东北风 微风'
}, 
    {
        'url': 'https://tianqi.fliggy.com/610100/', 
        'content': '果洛天气\n河池天气\n四平天气\n赣州天气\n呼和浩特天气\n嘉兴天气\n黔南天气\n镇江天气\n芜湖天气\n石家庄天气\n南通天气\n乐山天气\n池州天气\n南充天气\n长春天气\n鞍山天气\n盘锦天气\n李民烧鸡老店天气\n爱麦思休闲吧天气\n八旗乌拉满族火锅天气\n东山浴场天气\n雪列车天气\n乔一乔怪味餐厅(东升店)天气\n粗茶淡饭·汉口公馆天气\nGucci(成都天府机场店)天气\n小公牛酒楼天气\n重庆鼎味美蛙鱼头天气\n韩式旋转小火锅(金利来店)天气\n罗莎蛋糕(橙子街店)天气\n聚香乐酒楼天气\n先农坛天气\n食在湘寨天气\n港式烧腊天气\n食悦麻辣香锅天气\n\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n关注飞猪旅行天气助手 [...] 西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n西安 24 小时天气预报\n最近更新: 01:59\n0:00\n阴 [...] 2025年2月8日 星期六 农历正月十一最近更新: 01:59\n1℃最低 0℃ - 最高 5℃\n阴\n风向：东北风\n降水量：0.00mm\n风速：10km/h\n相对湿度：44%rh\n西安2月8日 01:59天气预报，今天0℃-5℃，阴, 风向情况：东北风，总降水量：0.00mm，相对湿度：44%。西安今日生活指数: 交通指数，良好 （天气较好，路面干燥，交通气象条件良好，车辆可以正常行驶。），旅游指数，适宜 （天气较好，但丝毫不会影响您出行的心情。温度适宜又有微风相伴，适宜旅游。）。未来5天，最低温度0℃，最高温度10℃，查询西安实时、5天天气预报，上飞猪天气预报频道\n2025年2月8日 星期六 农历正月十一最近更新: 01:59\n1℃最低 0℃ - 最高 5℃\n阴\n风向：东北风\n降水量：0.00mm\n风速：10km/h\n相对湿度：44%rh'
    }
]
```

### 4.5.7. 语法知识点

#### 4.5.7.1. langchain_core.runnables.Runnable

`langchain_core.runnables.Runnable` 是 LangChain 核心库中的一个重要抽象概念和基础组件，它为构建复杂的语言处理流程提供了强大且灵活的框架。

##### 4.5.7.1.1. 核心概念

`Runnable` 可被视作一个能够接收输入并产生输出的可调用对象。它能够代表各式各样的语言处理操作，像调用大语言模型（LLM）、执行文本转换、运用工具等。借助将不同的 `Runnable` 组合起来，我们能够构建出复杂的工作流，以完成诸如问答系统、文本生成、信息提取等任务。

##### 4.5.7.1.2. 主要特点

1. **可组合性**：`Runnable` 最大的亮点之一就是具备可组合性。我们能够把多个 `Runnable` 对象串联起来，构建出一个新的 `Runnable` 工作流。例如，我们可以先对输入文本进行预处理，接着调用大语言模型，最后对输出结果进行后处理。
2. **灵活性**：`Runnable` 能够代表各种类型的操作，既可以是简单的函数，也可以是复杂的类实例。这使得它在处理不同的任务和场景时具有很强的适应性。
3. **异步支持**：`Runnable` 支持异步操作，这意味着在处理大量数据或者进行耗时操作时，能够充分利用异步编程的优势，提高程序的性能和响应速度。

##### 4.5.7.1.3. 常见用法

###### 4.5.7.1.3.1. 定义单个 `Runnable`

我们可以通过定义一个函数或者类来创建一个 `Runnable`。以下是一个简单的示例，展示了如何定义一个将输入文本转换为大写的 `Runnable`：

```python
from langchain_core.runnables import Runnable

def to_uppercase(text):
    return text.upper()

uppercase_runnable = Runnable(to_uppercase)
result = uppercase_runnable.invoke("hello world")
print(result)  # 输出: HELLO WORLD
```

> 💡`.invoke()`可以理解为直接调用，类似于`__call__()`方法

###### 4.5.7.1.3.2. 组合多个 `Runnable`

我们可以使用 `|` 运算符将多个 `Runnable` 组合成一个新的 `Runnable`。以下是一个示例，展示了如何将两个 `Runnable` 组合起来：

```python
from langchain_core.runnables import Runnable

def add_prefix(text):
    return "Prefix: " + text

def to_uppercase(text):
    return text.upper()

prefix_runnable = Runnable(add_prefix)
uppercase_runnable = Runnable(to_uppercase)

combined_runnable = prefix_runnable | uppercase_runnable
result = combined_runnable.invoke("hello world")
print(result)  # 输出: PREFIX: HELLO WORLD
```

##### 4.5.7.1.4. 应用场景

- **构建复杂的语言处理工作流**：通过组合不同的 `Runnable`，我们可以构建出复杂的语言处理工作流，例如问答系统、文本生成系统等。
- **定制化处理**：我们可以根据具体需求定义自己的 `Runnable`，实现定制化的文本处理逻辑。

`langchain_core.runnables.Runnable` 为 LangChain 中的语言处理任务提供了一个灵活、可组合的框架，使得开发者能够更方便地构建和管理复杂的工作流。 

#### 4.5.7.2. langchain_core.messages.HumanMessage

`langchain_core.messages.HumanMessage` 是 LangChain 核心库中用于**表示人类用户输入消息的类**，在基于大语言模型（LLM）构建的对话系统或交互应用里发挥着关键作用。下面为我们详细介绍它的相关信息：

##### 4.5.7.2.1. 核心概念

在多轮对话的场景中，**需要清晰区分不同角色（如人类用户、AI 助手）的消息**。`HumanMessage` 类就是专门用来封装人类用户所输入的消息内容，以此为对话的上下文管理和消息传递提供支持。

##### 4.5.7.2.2. 主要特点

1. **明确角色标识**：借助 `HumanMessage` 类，能明确指出消息是由人类用户发出的，这有助于在对话历史中准确区分不同角色的发言，让对话的逻辑和流程更为清晰。
2. **与其他消息类型兼容**：`HumanMessage` 能够和 LangChain 中的其他消息类型（像 `AIMessage` 代表 AI 助手的回复、`SystemMessage` 代表系统级别的提示信息）协同工作，一同构建完整的对话历史。
3. **方便消息传递**：该类的实例可以方便地在不同组件（如消息处理器、大语言模型调用模块）之间传递，从而保证消息能够在整个对话流程中顺畅流转。

##### 4.5.7.2.3. 常见用法

以下是一个简单的示例，展示了如何创建 `HumanMessage` 实例并将其用于对话：

```python
from langchain_core.messages import HumanMessage

# 创建一个 HumanMessage 实例
human_input = "今天天气怎么样？"
human_message = HumanMessage(content=human_input)

# 打印消息内容
print(human_message.content)  # 输出: 今天天气怎么样？
```

##### 4.5.7.2.4. 应用场景

- **对话式 AI 系统**：在聊天机器人、智能客服等对话式 AI 系统中，`HumanMessage` 可用于封装用户的提问，然后将其与历史对话信息一同传递给大语言模型，以生成合适的回复。
- **多轮交互应用**：在需要多轮交互的应用（如交互式故事生成、智能问答游戏）中，`HumanMessage` 能记录用户每一轮的输入，为后续的交互和推理提供依据。

总之，`langchain_core.messages.HumanMessage` 为 LangChain 中处理人类用户输入消息提供了一种标准化且便捷的方式，有助于构建更加复杂和智能的对话系统。 

#### 4.5.7.3. .invoke('用户的输入')和.invoke([HumanMessage('用户的输入')])的区别

我们不禁有一个疑问：对于一个`Runnable`对象而言，`.invoke('用户的输入')`和`.invoke([HumanMessage('用户的输入')])`有什么区别？

在 LangChain 中，`Runnable` 对象的 `.invoke` 方法用于触发 `Runnable` 的执行，传入不同格式的参数（如普通字符串和 `HumanMessage` 对象列表）会有不同的处理方式，下面为我们详细分析两者的区别。

##### 4.5.7.3.1. 输入格式和语义

- **`.invoke('用户的输入')`**：
    - 这里传入的是一个普通的字符串。这种方式一般适用于那些期望接收简单文本输入的 `Runnable`，比如执行文本转换、简单的文本生成等任务。例如，我们有一个 `Runnable` 用于将输入文本转换为大写，那么直接传入字符串就能让它正常工作。
- **`.invoke([HumanMessage('用户的输入')])`**：
    - 传入的是一个包含 `HumanMessage` 对象的列表。`HumanMessage` 是 LangChain 中专门用于表示人类用户消息的类，使用这种方式传入输入，**更强调这是来自人类用户的消息，通常用于与对话相关的场景，比如对话式 AI 系统、聊天机器人等**。在这些场景中，系统需要区分不同角色（人类用户、AI 助手等）的消息，使用 `HumanMessage` 可以清晰地标识出消息的来源。

##### 4.5.7.3.2. 处理逻辑和适配场景

- **`.invoke('用户的输入')`**：
    - `Runnable` 对象会把传入的字符串当作原始输入进行处理，执行其内部定义的逻辑。例如，如果 `Runnable` 封装的是一个调用大语言模型进行文本生成的操作，它会直接将这个字符串作为提示词传递给大语言模型。
    - 这种方式适用于简单的、不需要考虑对话上下文和角色信息的任务，比如单纯的文本摘要、文本翻译等。
- **`.invoke([HumanMessage('用户的输入')])`**：
    - `Runnable` 对象会将 `HumanMessage` 对象作为一个完整的消息进行处理，可能会结合对话历史中的其他消息（如之前的 `HumanMessage` 和 `AIMessage`）来理解上下文。例如，在一个多轮对话的聊天机器人中，`Runnable` 会根据当前的 `HumanMessage` 和之前的对话历史来生成合适的回复。
    - 这种方式适用于需要处理复杂对话场景、维护对话上下文的任务，比如智能客服系统、交互式故事生成等。

##### 4.5.7.3.3. 示例代码说明

下面是两个简单的示例，展示了两种调用方式的不同应用场景：

```python
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

# 示例 1: 简单文本处理，使用普通字符串输入
def to_uppercase(text):
    return text.upper()

uppercase_runnable = Runnable(to_uppercase)
result1 = uppercase_runnable.invoke('hello world')
print(result1)  # 输出: HELLO WORLD

# 示例 2: 模拟对话场景，使用 HumanMessage 输入
# 这里只是简单模拟，实际中会更复杂
def chat_response(messages):
    human_message = messages[0]
    return f"你说: {human_message.content}"

chat_runnable = Runnable(chat_response)
human_msg = [HumanMessage(content='今天天气怎么样？')]
result2 = chat_runnable.invoke(human_msg)
print(result2)  # 输出: 你说: 今天天气怎么样？

```

在示例 1 中，`Runnable` 只需要简单的文本输入就能完成任务；而在示例 2 中，使用 `HumanMessage` 可以更好地模拟对话场景，让 `Runnable` 能够处理对话相关的逻辑。 


## 4.6. 案例五：Langchain构建RAG的对话应用

### 4.6.1. 案例介绍

本案例是：复杂的问答 (Q&A) 聊天机器人。应用程序可以回答有关特定源信息的问题。使用一种称为检索增强生成 (RAG) 的技术。

> RAG是一种增强大型语言模型（LLM）知识的方法，它通过引入额外的数据来实现。

### 4.6.2. 实现思路

- **加载**：首先，我们需要加载数据。这是通过`DocumentLoaders`完成的。
- **分割**： `Text splitters`将大型文档分割成更小的块。这对于索引数据和将其传递给模型很有用，因为大块数据更难搜索，并且不适合模型的有限上下文窗口。
- **存储**：我们需要一个地方来存储和索引我们的分割，以便以后可以搜索。这通常使用`VectorStore`和`Embeddings`模型完成。
- **检索**：给定用户输入，使用检索器从存储中检索相关分割。
- **生成**：`ChatModel `/ `LLM`使用包括问题和检索到的数据的提示生成答案。

### 4.6.3. 代码

```python
import os
import bs4
import langchain
import langchain.chains
import langchain_community
import langchain_core
import langchain.chains.combine_documents
import langchain.chains.history_aware_retriever
import langchain.chains.retrieval
import langchain_community.chat_message_histories
import langchain_core.runnables
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 声明环境变量
os.environ['LANGCHAIN_TRACING_V2'] = 'true'  # 声明LangChain的版本

def get_session_history(session_id: str):
    """
    此函数的作用是根据给定的 session_id 获取对应的会话历史记录。
    如果指定的 session_id 不存在于 vector_store 中，会为该 session_id 创建一个新的会话历史记录。

    Args:
        session_id (str): 用于标识特定会话的唯一字符串。

    Returns:
        ChatMessageHistory: 对应 session_id 的会话历史记录对象。
    """
    # 检查传入的 session_id 是否不在 vector_store 这个数据结构中
    if session_id not in vector_store:
        # 如果 session_id 不在 vector_store 中，为该 session_id 创建一个新的 ChatMessageHistory 对象
        # 并将其存储在 vector_store 中，键为 session_id
        vector_store[session_id] = langchain_community.chat_message_histories.ChatMessageHistory()
    # 返回 vector_store 中对应 session_id 的会话历史记录对象
    return vector_store[session_id]

if __name__ == '__main__':
    # 创建文档加载器的类
    content_loader = WebBaseLoader(
        web_path=[
            'https://lilianweng.github.io/posts/2023-06-23-agent/',
        ],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=(  # 指定哪些元素是需要读取的
                    'post-header',
                    'post-title',
                    'post-content'
                )
            )
        )
    )

    # 正式加载文档 -> 一篇博客内容的博客
    docs: list = content_loader.load()

    # 大文本的切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个文本块的最大字符数量。文本会按这个大小进行分割，从而生成多个文本块
        chunk_overlap=200  # 相邻文本块之间重叠的字符数量。其作用是保证文本块之间有一定的连续性，防止关键信息在分割时被截断
    )
    # 正式切割
    splits: list = splitter.split_documents(documents=docs)

    # 存储：将文本转换为向量
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(base_url="https://www.dmxapi.com/v1"),
    )

    # 创建检索器
    retriever = vector_store.as_retriever()

    # 创建问题的Prompt模板
    system_prompt = """你是负责问答任务的助手。请使用检索到的上下文信息回答问题。
    若无法确定答案，请直接说明未知。
    答案需简洁，最多三句话。\n
    
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ('system', system_prompt),  # LLM初始化时看到的Prompt
            MessagesPlaceholder('chat_history'),  # 历史聊天记录（这里用占位符先占住）
            ('human', "{input}")  # 用户输入的Prompt
        ]
    )

    # 构建LLM模型
    model: Runnable = ChatOpenAI(
        model='gpt-3.5-turbo',
        base_url="https://www.dmxapi.com/v1"
    )

    # 创建文档组合链
    document_combine_chain: Runnable = langchain.chains.combine_documents.create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    # 创建子链
    ## 子链的Prompt Template
    contextualize_q_system_prompt: str = """给定一段聊天记录和用户的最新问题（该问题可能涉及聊天记录中的上下文），
    请构造一个无需依赖聊天记录即可理解的独立问题。
    请勿回答该问题，仅在需要时对其进行重新表述，否则直接按原样返回。"""

    retriever_history_template = ChatPromptTemplate.from_messages(
        messages=[
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    ## 创建历史记录链
    history_chain = langchain.chains.history_aware_retriever.create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_history_template
    )

    # 保存聊天的历史记录
    vector_store: dict = {}

    # 创建终极链: 将前面创建的两个链整合在一起
    chain = langchain.chains.retrieval.create_retrieval_chain(
        retriever=history_chain,
        combine_docs_chain=document_combine_chain
    )

    result_chain = langchain_core.runnables.RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key='input',  # 用户输入的key
        history_messages_key='chat_history',  # 历史记录的key
        output_messages_key='answer'  # 返回结果的key
    )

    # 第一轮对话
    resp_1 = result_chain.invoke(
        input={'input': '什么是任务分解？'},
        config={'configurable': {'session_id': 'Le0v1n_001'}}
    )
    print(f"---------- 第一轮对话 ----------\n{resp_1['answer']}\n")

    # 第二轮对话 -> 考验是否有上下文能力
    resp_2 = result_chain.invoke(
        input={'input': '它有哪些通用的方法呢？'},
        config={'configurable': {'session_id': 'Le0v1n_001'}}  # 这里的session_id不能变
    )
    print(f"---------- 第二轮对话 ----------\n{resp_2['answer']}\n")

    # 额外对话（修改了会话id -> 没有上下文能力）
    resp_3 = result_chain.invoke(
        input={'input': '它有哪些通用的方法呢？'},
        config={'configurable': {'session_id': 'Le0v1n_002'}}  # 这里的session_id不能变
    )
    print(f"---------- 额外对话 ----------\n{resp_3['answer']}\n")
```

```
---------- 第一轮对话 ----------
任务分解是将复杂任务拆分成多个更小、更简单的步骤的过程。它可以通过不同的方法实现，如使用提示给LLM、特定任务的指令或人类输入。任务分解有助于提升模型在复杂任务中的表现，常见技术包括链式思维（CoT）和思想树（Tree of Thoughts）。

---------- 第二轮对话 ----------
任务分解的通用方法包括：

1. 使用简单提示，例如“XYZ的步骤是？”或“实现XYZ的子目标有哪些？”来引导大型语言模型（LLM）。
2. 采用特定任务的指令，如“为小说撰写故事大纲”。
3. 通过人类输入提供直接的分解指导。

---------- 额外对话 ----------
一些通用的方法包括任务分解（可以通过简单的提示、任务特定指令或人类输入实现）、树状思维结构（如Tree of Thoughts，探索多个推理可能性），以及利用API来增强LLM的功能（如API-Bank Benchmark）。这些方法旨在优化LLM的推理和任务执行能力。
```

## 4.7. 案例六：Langchain读取数据库

### 4.7.1. 案例介绍

本案例介绍在数据库中表格数据上的问答系统的基本方法。将涵盖使用链（chains）和代理（agents）的实现。通过查询数据
库中的数据并得到自然语言答案。两者之间的主要区别在于，我们的代理可以根据需要多次循环查询数据库以回答问题。

### 4.7.2. 实现思路

1. **将问题转换为DSL查询**：模型将用户输入转换为SQL查询
2. **执行SQL查询**：执行查询。
3. **回答问题**：模型使用查询结果响应用户输入。

<div align=center>
    <img src=./imgs_markdown/2025-04-08-11-38-20.png
    width=70%></br><center></center>
</div>

> 🤔 𝑸𝒖𝒆𝒔𝒕𝒊𝒐𝒏：什么是DSL？
>
> 🥳 𝑨𝒏𝒔𝒘𝒆𝒓：在数据库领域，**DSL（Domain-Specific Language，领域特定语言）** 是专门为高效处理数据库任务设计的简洁语言，只聚焦数据库相关操作（如查询、管理数据）。
> 
> 最典型的例子是 **SQL**，它是关系型数据库的标准DSL，用于查询数据（如 `SELECT`）、增删改数据（如 `INSERT/UPDATE/DELETE`）、定义表结构（如 `CREATE TABLE`）等。
> 
> DSL的特点是 **声明式语法**：用户只需描述“需要什么”（如“查年龄大于30的用户”），无需关心“如何实现”，数据库会自动优化执行。
> 
> 简单说，DSL就是为数据库定制的“专用工具”，让操作更简单、高效。

### 代码

#### SQLAlchemy

我们需要考虑一个问题，我们应该如何连接数据库，这里Langchain使用的是`SQLAlchemy`这个包来进行的。

##### 支持的数据库

URL通常可以包括用户名、密码、主机名、数据库名以及用于其他配置的可选关键字参数。主题格式为：

```bash
dialect+driver://username:password@host:port/database
```

###### SQLite数据库

sqlite使用python内置模块连接到基于文件的数据库sqlite3默认情况下。

```python
# Unix/Mac - 4 initial slashes in total
engine = create_engine('sqlite:////absolute/path/to/foo.db')

# Windows
engine = create_engine('sqlite:///C:\\path\\to\\foo.db')

# Windows alternative using raw string
engine = create_engine(r'sqlite:///c:\path\to\foo.db')

#当前目录下的test.db
engine = create_engine(r'sqlite:///test.db' echo=True, future=True)
```

###### MySQL数据库

mysql方言使用mysql python作为默认dbapi。mysql dbapis有很多，包括`pymysql`和`mysqlclient`:

```python
# default（这个默认我们不用）
engine = create_engine('mysql://scott:tiger@localhost/foo?charset=utf8')

# mysqlclient(a maintained fork ofMysQL-Python)
engine = create_engine('mysql+mysqldb://scott:tiger@localhost/foo?charset=utf8')

# PyMySQL
engine = create_engine('mysql+pymysql://scott:tiger@localhost/foo?charset=utf8')
```

###### Microsoft SQL数据库

SQL Server方言使用pyodbc作为默认`dbapi`。PYMSSQL也可用：

```python
# pyodbo
engine = create_engine('mssql+pyodbc://scott:tiger@mydsn')

# pymssql
engine = create_engine('mssql+pymssql://scott:tiger@hostname:port/dbname')
```

#### 实现代码

```python
```





https://www.bilibili.com/video/BV1g1CKYUExu?spm_id_from=333.788.player.switch&vd_source=c96423a5c3cff0232795a1e42972c3d4&p=20

