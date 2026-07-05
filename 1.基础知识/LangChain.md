
## 为什么需要？

**为什么需要大模型应用开发框架?**
大模型应用开发的痛点
- 基础调用局限性：大语言模型仅提供基础调用方式，无法满足复杂应用需求
- 工程问题清单：
    - 上下文记忆缺失：模型默认无记忆功能，无法关联多轮对话
    - 外部能力缺失：无法进行网络检索或加载本地数据
    - Prompt管理困难：每次需完整输入prompt，缺乏便捷管理方式
    - 模型切换成本高：不同模型输入输出差异大，需大量代码修改

## 基础架构

```
LangChain 6大核心组件
│
├── Models（模型）
│   ├── LLMs（大语言模型）
│   ├── Chat Models（聊天模型）
│   └── Text Embedding Models（文本嵌入模型） ────── 70+个大语言模型 / 10+个文本嵌入模型
│
├── Prompts（提示）
│   └── Prompt Templates（提示模板） ────────────── 5+种输出解析
│
├── Indexes（索引）
│   ├── Document Loaders（文档加载器）
│   ├── Text Splitters（文本分割器）
│   ├── Vector Stores（向量存储）
│   └── Retrievers（检索器） ────────────────────── 50+种文档加载器 / 10+种向量存储
│
├── Memory（记忆）
│   ├── Chat Message History（聊天消息历史）
│   └── 长上下文总结能力
│
── Chains（链）
│   ├── Chain（基础链）
│   ├── LLM Chain（大语言模型链）
│   ├── Index-related Chains（检索链）
│   └── Custom Chain（自定义链） ────────────────── 20+种链类型
│
└── Agents（代理）
    ├── Tool（工具）
    ├── Agents（代理）
    ├── Toolkits（工具箱）
    └── Agent Executor（代理执行者） ─────────────── 5+种代理类型 / 70+个工具箱
```

### Prompts
![[images/Pasted image 20260704093534.png]]
![[images/Pasted image 20260704093705.png]]

不同 Prompt 组件功能的简介： 
* PromptTemplate：用于创建文本消息提示模板，用于用于与大语言模型/文本生成模型进行交 互。 
* ChatPromptTemplate：用于创建聊天消息提示模板，一般用于与聊天模型进行交互。
* MessagePlaceholder：消息占位符，在聊天模型中对不确定是否需要的消息进行占位。
* SystemMessagePromptTemplate：用于创建系统消息提示模板，角色为系统。
* HumanMessagePromptTemplate：用于创建人类消息提示模板，角色为人类。
* AIMessagePromptTemplate：用于创建AI消息提示模板，角色为AI。
* PipelinePromptTemplate：用于创建管道消息，管道消息可以将提示模板作为变量进行快速复 用。 

Prompt 不同方法的功能简介： 
* partial：用于格式化提示模板中的部分变量。 
* format：传递变量数据，格式化提示模板为文本消息。 
* invoke：传递变量谁，格式化提示模板为提示。 to_string：将提示/消息提示列表转换成字符串。 
* to_messages：用于将消息提示列表转换成字符串。 

Prompt 中重载的运算符： +运算符：
在 Prompt 组件中，对 + 运算符使用 __add__ 方法进行重写，所以几乎所有 Prompt 组件都可以使用 + 进行组装拼接

### Model
![[images/Pasted image 20260704110902.png]]

**LangChain 为两种类型的模型提供接口和集成：** 
* LLM：使用纯文本作为输入和输出的大语言模型。 
* Chat Model：使用聊天消息列表作为输入并返回聊天消息的聊天模型。

**调用大模型最常用的方法为：** 
1. invoke：传递对应的文本提示/消息提示，大语言模型生成对应的内容。 
2. batch：invoke 的批量版本，可以一次性生成多个内容。 
3. stream：invoke 的流式输出版本，大语言模型每生成一个字符就返回一个字符
![[images/Pasted image 20260704111135.png]]
![[images/Pasted image 20260704113133.png]]
### Memory


### Chain

### Agents

## langchainCore

###  OutputParser组件

输出解析器 = 预设提示 + 解析功能 

在 LangChain 中，输出解析器通常包含两个抽象函数的实现，这也是自定义输出解析器需要实现的两个函数：
1. get_format_instructions ：用来约定输出的格式，并转换为描述文本。
2. parse ：用来解析 LLM 的输出为约定的格式。

![[images/Pasted image 20260704171913.png]]

### LCEL表达式与Runnable可运行协议

#### 多组件Invoke嵌套的缺点
1. 可阅读性降低
2. 难以排查（无法得知每一步的具体结果与执行进度
3. 无法继承大量组件

解决：嵌套的写法 改成 **平级调用**

prompt、model、outparser都有一个共同调用方法invoke，所以组装后依次调用

 #### 手写一个"Chain"示例
```python
from typing import Any
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
parser = StrOutputParser()

class Chain: steps: list = []
	def __init__(self, steps: list): self.steps = steps
	def invoke(self, input: Any) -> Any: output: Any = input
	for step in self.steps: 
		output = step.invoke(output)
		print(step)
		print("执行结果:", output)
		print("===============")
	return output

chain = Chain([prompt, llm, parser])

print(chain.invoke({"query": "你好，你是?"}))

```
#### Runnable与LCEL表达式