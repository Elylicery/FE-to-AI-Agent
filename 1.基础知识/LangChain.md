
## 1. 介绍
## 1.1 背景

**为什么需要大模型应用开发框架?**
大模型应用开发的痛点
- 基础调用局限性：大语言模型仅提供基础调用方式，无法满足复杂应用需求
- 工程问题清单：
    - 上下文记忆缺失：模型默认无记忆功能，无法关联多轮对话
    - 外部能力缺失：无法进行网络检索或加载本地数据
    - Prompt管理困难：每次需完整输入prompt，缺乏便捷管理方式
    - 模型切换成本高：不同模型输入输出差异大，需大量代码修改

###  1.2 基础架构

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

#### Prompts
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

#### Model
![[images/Pasted image 20260704110902.png]]

**LangChain 为两种类型的模型提供接口和集成：** 
* LLM：使用纯文本作为输入和输出的大语言模型。 
* ChatModel：使用聊天消息列表作为输入并返回聊天消息的聊天模型。

**调用大模型最常用的方法为：** 
1. invoke：传递对应的文本提示/消息提示，大语言模型生成对应的内容。 
2. batch：invoke 的批量版本，可以一次性生成多个内容。 
3. stream：invoke 的流式输出版本，大语言模型每生成一个字符就返回一个字符
![[images/Pasted image 20260704111135.png]]
![[images/Pasted image 20260704113133.png]]
#### Indexes
见后文
#### Memory
见后文

#### Chain
见后文
#### Agents
见后文

## 2. langchainCore

###  OutputParser组件

输出解析器 = 预设提示 + 解析功能 

在 LangChain 中，输出解析器通常包含两个抽象函数的实现，这也是自定义输出解析器需要实现的两个函数：
1. get_format_instructions ：用来约定输出的格式，并转换为描述文本。
2. parse ：用来解析 LLM 的输出为约定的格式。

![[images/Pasted image 20260704171913.png]]

### LCEL表达式与Runnable可运行协议

**多组件Invoke嵌套的缺点**
1. 可阅读性降低
2. 难以排查（无法得知每一步的具体结果与执行进度
3. 无法继承大量组件

**解决：**嵌套的写法 改成 **平级调用**

prompt、model、outparser都有一个共同调用方法invoke，可以组装后依次调用

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

![[images/Pasted image 20260705094056.png]]
![[images/Pasted image 20260705094127.png]]

上面Chain的写法等价下面
```python
chain = prompt | llm | parser
```
Runnable 底层的运行逻辑本质上也是将每一个组件添加到列表中，然后按照顺序执行并返回最终结果， 核心源码
```python
def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
	from langchain_core.beta.runnables.context import config_with_context
\
	# setup callbacks and context 
	config = config_with_context(ensure_config(config), self.steps) 
	callback_manager = get_callback_manager_for_config(config)
	# start the root run
	run_manager = callback_manager.on_chain_start(
		dumpd(self),
		input, 
		name=config.get("run_name") or self.get_name(), 
		run_id=config.pop("run_id", None),
	)
	
	# 调用所有步骤并逐个执行得到对应的输出，然后作为下一个的输入
	try:
		for i, step in enumerate(self.steps):
			input = step.invoke(
				input, 
				# mark each step as a child run 
				patch_config( config, callbacks=run_manager.get_child(f"seq:step: {i+1}")),
			)
	# finish the root run
	except BaseException as e:
		run_manager.on_chain_error(e)
		raise
	else:
		run_manager.on_chain_end(input)
		return cast(Output, input)

```

### 两个Runable核心类

**RunnableParallel 并行运行：**

支持运行多个 Runnable 的类，一般用于操作 Runnable 的输 出，以匹配序列中下一个 Runnable 的输入，起到并行运行 
Runnable 并格式化输出结构的作用。 

例如 RunnableParallel 可以让我们同时执行多条 Chain，然后以字典的形式返回各个 Chain 的结果，对 比每一条链单独执行，效率会高很多

```python
chain = RunnableParallel( context=retrieval,query=RunnablePassthrough(), ) | prompt | llm | parse
```

**RunablePassthrough 传输数据**

RunnablePassthrough，这个类透传上游参数输入，简单来说，就是可以获取上游的数据，并保持不变 或者新增额外的键。 通常与 RunnableParallel 一起使用，将数据分配给映射中的新键
```python
chain = {"query": RunnablePassthrough()} | prompt | llm | StrOutputParser()
```

### 利用回调功能调式链应用

**Callback**
![[images/Pasted image 20260705102151.png]]

**CallbackHandler**：对每个应用场景比如 Agent 或 Chain 或 Tool 的纪录。 


**langSmith调试平台** 

LangSmith 是 LangChain 生态里的可观测性平台，主要解决三件事：Trace（追踪每次 LLM 调用的完整链路，包括 prompt、token 消耗、延迟、中间步骤）、Evaluate（批量跑测试集，评估 Agent 或 RAG 的输出质量）、Prompt Management（版本管理和 A/B 测试 prompt）

## 3. LLM记忆模块

### 3.1 记忆功能 

#### 3.1.1 **如何为LLM应用添加记忆功能？**

- 核心机制：通过外部模块保存对话上下文信息，在每次请求时将历史信息作为输入传递给模型
- 实现方法：在Prompt中预留`chat_historychat`占位符，实时保存并插入`Human/AI`对话信息
- 流程特点：
    - **模型本身不具备记忆能力，完全依赖用户输入产生输出**
    - 需要额外开发存储模块（内存/数据库/本地文件）保存对话记录
    - 每次对话时将完整历史记录插入Prompt模板
![[images/Pasted image 20260706222533.png]]

记忆功能集成LLM应用的两个核心问题：存储的历史信息是什么？如何检索与处理历史信息

##### ChatMessageHistory

langChain的聊天消息历史存储组件：**ChatMessageHistory组件**，即“消息容器”
![[images/Pasted image 20260707220037.png]]

#### 3.1.2 常见记忆模式

##### 缓冲记忆

>  本质：保留最近“N条“消息
  
* ***缓冲记忆**：全量保留：存储所有Human/AI生成的消息，使用时将完整聊天历史传递到Prompt中
* **缓冲窗口记忆**：按条数截断：设置窗口值k(如k=2)，只保留最近k次互动
* **令牌缓冲记忆**：按token数阶段：用max_tokens替代固定窗口值k

##### 摘要总结记忆
对历史对话进行压缩摘要而非完整存储

![[images/Pasted image 20260706223811.png|585]]

##### ⭐摘要缓冲混合记忆
> 工程上最实用的方案，长期为模糊记忆，短期为精确记忆

前面用摘要、后面保留最近几轮原文，兼顾了上下文完整性和 token 控制
![[images/Pasted image 20260706224154.png]]

实现所需模块：
    - ﻿chat_message_history：存储历史对话消息列表
    - ﻿moving_summary_buffer 存储被移除消息的汇总字符串
    - ﻿summary_llm：接收当前摘要、用户提问和AI回复生成新摘要的模型
    - ﻿max_tokens：设置记忆模块存储的最大token数（本案例设为300）
    - ﻿get_num_tokens：统计文本token数量的函数
工作流程：
    - 当历史消息token数超过max_tokens时触发清理
    - 移除最旧的一组human-AI对话
    - 将被移除的对话内容生成摘要存入moving_summary_buffer
    - 循环执行直到token数低于阈值
#####  向量存储库记忆

> 用得少，长对话场景的解法，一般用RAG

使用向量数据库存储对话片段，通过相似度检索相关记忆（实现复杂: 需要Embedding+向量数据库支持）

![[images/Pasted image 20260706224357.png]]

### 3.2 LangChain记忆组件
#### 3.2.1 记忆分类
![[images/Pasted image 20260707220125.png|602]]

##### 缓冲记忆组件

**LangChain中最简单的记忆组件，采用"原进原出"机制，不对数据结构和提取算法做额外处理**

* conversationBufferMemory：完整保存所有对话历史（Human和AI的交互记录）
* ConversationBufferwindowMemory：通过`k`值控制记忆窗口大小，保留最近`2∗k`条对话
* ConversationTokenBufferMemory：通过`max_token_limits`设置令牌数上限

##### 摘要记忆组件

方案: 保留关键信息（重点记忆），移除冗余噪音（流水式信息）
 * conversationSummaryMemory：摘要总结记忆，**将传递的历史对话记录总结成摘要进行保存**
 * conversationSummaryBufferMemory：摘要缓冲混合记忆，**在不超过max_token_limit的限制下，保留对话历史数据，对于超过的部分，进行信息的总结与提取**

#### 3.2.2  Memory组件运行流程

![[images/Pasted image 20260707220140.png]]

##### RunnableWithMessageHistory

RunnableWithMessageHistory 把一个普通的Runnable（比如一个Chain和Model）和消息历史粘合在一起。
核心作用是：在每次调用之前自动从存储中加载历史消息并注入到链的输入中，在调用结束后自动把新的消息写入存储。
它解决的是"如何让一个无状态的链拥有多轮对话记忆"这个问题。是一个存取消息的机制。

运行流程
![[images/Pasted image 20260708172810.png]]
代码示例
```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 创建消息存储（底层容器）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 2. 用 RunnableWithMessageHistory 包装链
model = ChatOpenAI()

chain_with_history = RunnableWithMessageHistory(
    model,
    get_session_history,       # 告诉它去哪里存取消息
    input_messages_key="input",
    history_messages_key="history",
)

# 3. 调用时传入 session_id，它会自动处理历史消息的读写
chain_with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"session_id": "user_123"}},
)
```
#### 3.3 记忆组件持久化与Chain组件扩展

核心思路：通过外部存储实现记忆持久化，将记忆组件数据存储在文件或数据库中，使用时重新实例化
![[images/Pasted image 20260707225059.png]]
集成接入的第三方对话历史消息有：postsqlres、sqlite、redis、kalfa等。。。并且chatMessageHistory可以接入到Runnable可运行协议的链条 

![[images/Pasted image 20260707225518.png]]

#### 3.3.4 开源智能体记忆模块解析
## 4. Runable可运行协议深入
### 4.1.  Runable配置方法

**bind函数**：通过`Runnable.bind()`方法可以为Runnable组件添加默认调用参数，可以用来给 Runnable 做参数预设，让链式调用更简洁、配置更可复用。


configurable_fields:  运行时动态修改链的参数，使用时分成两个流程：
1. 为Runable定义哪些字段可以在链运行时动态配置
2. 在调用invoke()时，传递对应的配置信息confugurable完成动态配置

![[images/Pasted image 20260709224548.png]]

configurable_alternatives：实现运行时组件替换（例如在**构建的链应用中，动态替换掉特定的模型、提示词等整个组件本身，而不是替换组件里的参数信息**）
![[images/Pasted image 20260709225252.png]]

