  
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

LangChain核心概念
1.LLM
2.PromptTemplate
3.Chain（链）
4.Agent（代理）
5.Memory
6.Document Loader & Text Splitter
7.Embedding Model
8.Vector Store

langchain六大模块
1.模型I/O模块
2.RAG模块
3.存储模块
4.工具模块
5.回调
6.LCEL语法

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
#### 4.1.1  bind 动态绑定组件默认参数

**bind函数**：通过`Runnable.bind()`方法可以为Runnable组件添加默认调用参数，可以用来给 Runnable 做参数预设，让链式调用更简洁、配置更可复用。

#### 4.1.2 configurable_fields 动态绑定运行时参数
configurable_fields:  运行时动态修改链的参数，使用时分成两个流程：
1. 为Runable定义哪些字段可以在链运行时动态配置
2. 在调用invoke()时，传递对应的配置信息confugurable完成动态配置

![[images/Pasted image 20260709224548.png]]

#### 4.1.3 configurable_alternatives 动态替换运行时组件
configurable_alternatives：实现运行时组件替换（例如在**构建的链应用中，动态替换掉特定的模型、提示词等整个组件本身，而不是替换组件里的参数信息**）
![[images/Pasted image 20260709225252.png]]

### 4.2 组件重试与回退机制

#### 4.2.1 with_retry 组件重试

with_retry 通过构建一个新的Runable，在执行调用类的函数时，循环特定次数，直到组件能正常执行结束即暂停，并且在每次循环的过程中，休眠特定的时间
![[images/Pasted image 20260712094734.png]]

#### 4.2.2 with_fallback 回退机制
with_fallback：当组件出错时不重试，而是执行备用方案

![[images/Pasted image 20260712095202.png]]
### 4.3 组件生命周期

#### 4.3.1 with_listeners 生命周期监听
with_listeners为开始、结束、出错这3个常见的生命周期进行监听，底层是使用callbackdhandler实现对应的监听
![[images/Pasted image 20260712103434.png]]

### 4.4 封装记忆链

通过config+configurable形式传递memory实例，在链的执行函数中通过第二个参数获取memory实例
![[images/Pasted image 20260712103627.png]]

### 4.5 多智能体记忆模块解读

**开源智能体MetaGPT解读**

MetaGPT 底层是通过多个 Agent 之间相互协作, 相互共享信息来完成复杂项目的编写, 由于 LLM 上下文长度的限制, MetaGPT 必须通过外部的记忆机制来实现对 Agent 交流对话之间以及历史对话信息/知识的管理。

MetaGPT 记忆模块相关的功能, 全部存储到 memory 文件夹下, 涵盖了 记忆基础类、长期记忆、记忆存储库、记忆大脑 等,类图如下

![[images/Pasted image 20260712111438.png]]

**其他优秀Agent项目**

* AutoGPT
* babyAGI
* AutoGen

## 5. LangChain中的Embedding

### 5.1 Embedding组件

enbedding类的方法：
1. embed_documents：嵌入文档列表
2. embed_query：嵌入单个查询，字符串对应向量 

### 5.2 CacheBackEmbedding组件

cacheBackEmbedding类，可以使用缓存数据，避免对重复数据的二次计算，提高响应速度。

其运行流程本质上是封装了一个持久化存储的数据存储仓库，每次数据嵌入钱，先从数据存储仓库检索对应向量，没找到再进行文本调用嵌入向量

## 6. RAG组件
### 6.1 Document

Document类定义了一个文档对象的结构，涵盖了文本内容和相关的元数据，**Document也是文档加载器、文档分割器、向量数据库、检索器这几个组件之间交互传递的状态数据**

```
﻿Document=page_content(页面内容)+metadata(元数据)
```

RAG开发流程：
数据来源本地markdown/HTML/PDF/DOC文件或URL等。在RAG应用外部，一般会有一个额外的扩展，用来处理**读取数据→切割文档→存储到向量数据库**（该流程非常耗时，一般使用队列/异步进行处理）

架构流程图
![[images/Pasted image 20260716211843.png]]
### 6.2 文档加载器

#### 6.2.1 内置文档加载器

##### 文本加载器TextLoader

可以加载一个文本文件（code、markdown、text等，doc不是文本文件），把文件的内容读入到Document对象中，底层是 open(file_path, encoding=encoding) 读文本文件
```python
#langchain_community/document_loaders/text.py 的核心
def load(self) -> List[Document]:
    with open(self.file_path, encoding=self.encoding) as f:
        text = f.read()
    metadata = {"source": self.file_path}
    return [Document(page_content=text, metadata=metadata)]
```

##### 其他高频内置文档加载器

其他高频内置文档加载器：Markdown文档加载器、office（doc、xlsx、ppt）文档加载器、URL网页加载器（webBaseLoader）

langchain中的文档加载器
![[images/Pasted image 20260716212129.png]]

每一个文档加载器的使用流程都是：
1. 传递对应的参数实例化
2. 调用加载器的`load()` 方法加载文档 

#### 6.2.2 非架构化加载器
##### 通用文件加载器的使用技巧

实际LLM应用开发中，由于数据的种类是无穷的，对于一些无法判断的数据类型或者是通用性文件加载，可以使用**unstructedFileLoader非结构化文件加载器**来实现对文件的加载

（底层使用开源`unstructure`包处理非结构化数据，是langchain文档加载器的核心）
```python
if file_extension in [".xlsx", ".xls"]:
    loader = UnstructuredExcelLoader(file_path)
elif file_extension == ".pdf":
    loader = UnstructuredPDFLoader(file_path)
elif file_extension in [".md", ".markdown"]:
    loader = UnstructuredMarkdownLoader(file_path)
elif file_extension in [".htm", "html"]:
    loader = UnstructuredHTMLLoader(file_path)
elif file_extension in [".docx", ".doc"]:
    loader = UnstructuredWordDocumentLoader(file_path)
elif file_extension == ".csv":
    loader = UnstructuredCSVLoader(file_path)
elif file_extension in [".ppt", ".pptx"]:
    loader = UnstructuredPowerPointLoader(file_path)
elif file_extension == ".xml":
    loader = UnstructuredXMLLoader(file_path)
else:
    loader = UnstructuredFileLoader(file_path) if is_unstructured else TextLoader(file_path)
```

#### 6.2.3 自定义文档加载器

对于数据库、API接口等定制化强的企业数据，通用加载器可能无法满足格式需求。例如使用WebBaseLoader加载网页会提取大量空白数据（空格/换行/Tab），影响向量数据库的检索效率和准确性。所以需要自定义文档加载器

**实现：继承BaseLoader基类，实现lazy_load()方法**
（如果需要异步使用还要实现alazy_load()方法）

#### 6.2.4 Blob加载器
##### 文档加载器的核心流程

lazy_load() 方法的核心数据：读取文件数据 -》 将文件数据解析成Document

所以：**文档加载器 = 二进制数据读取+ 解析逻辑**

文档加载器运行流程：
![[images/Pasted image 20260716220533.png]]

##### Blob与BlobParse代替文档加载器

* **背景**：许多文档加载器都涉及文件解析，差异主要源于解析方式而非加载方式。
* **解决方案**：受Web Blob API启发，LangChain提供Blob方案来处理二进制数据转换。
- **核心组件**:
    - Blob: 封装的数据对象，表示原始数据
    - BlobLoader: 数据加载器，设计目标是支持任意数据加载
    - BlobParser: 数据解析器，将Blob转换为文档列表

Blob方案的文档加载器运行流程:
![[images/Pasted image 20260716220859.png]]

### 6.3 文档转换器 

#### 6.3.1 分割类的DocumentTransformaer组件

使用文档加载器得到的问题存在 **原始文档太大、原始文档数据格式不符合需求、原始温昂信息没有经过提炼等问题**

如果将这类数据直接转换成向量并存储到数据库中，会导致在执行相似性搜索和 RAG 的过程中，错误率大大提升。所以在 在加载完数据后，一般会执行多一步 **转换** 的过程，即将加载得到的 **文档列表** 进行转换，得到符合需求的 **文档列表** 。

转换涵盖的操作就非常多，例如：文档切割、文档属性提取、文档翻译、HTML 转文本、重排、元数据标记等都属于转换。
![[images/Pasted image 20260719101247.png]]


langchain中的组件分类：文档分割器、文档的处理转换（老版本写法。基类是BaseDocumentTransformer

![[images/Pasted image 20260719101428.png]]

#### 6.3.2 文档分割器
##### 字符分割器

文档分割器中最简单的—— **字符串分割器**CharacterTextSplitter，基于给定的字符串进行分割，默认为\n\n，并且在分割时会尽可能保证数据的连续性。

```python

loader = unstrcutedMarkdownLoader("./test.md
")
documents= loader.load()

text_spliter = CharacterTextSplitter()
chunks = text_spltter.split_document(docuemnts)
```

##### ⭐递归字符文本分割器

**递归字符文本分割器**

普通的字符文本分割器只能使用单个分隔符对文本内容进行划分，在划分的过程中，可能会出现文档块` 过小` 或者` 过大` 的情况，这会让 `RAG` 变得不可控，例如：

1. **文档块可能会变得非常大**，极端的情况下某个块的内容长度可能就超过了 LLM 的上下文长度限制
    
2. **文档块可能会远远小于窗口大小**，导致文档块的信息密度太低  

**分割思路：按照 `分隔符` 初次分割的时候，去检测块内容，如果太大就按照提供的 `备选分隔符` 二次分割，如果太小则合并前后的块**，最后让所有的块内容长度都控制在指定的大小并尽可能接近


**LangChain中的解决方案：**
**RecursiveCharacterTextSplitter 递归字符串分割**，这个分割器可以传递 一组分隔符 和 设定块内容大小，根据分隔符的优先顺序对文本进行预分割，然后将小块进行合并，将大块进行递归分割，直到获得所需块的大小，最终这些文档块的大小并不能完全相同，但是仍然会逼近指定长度。

`RecursiveCharacterTextSplitter`的分隔符参数默认为 `["\n\n", "\n", " ", ""]`(使用顺序：两行的数据 -> 单个换行符 -> 空格 ->单个字符，更符合英文习惯

**RecursiveCharacterTextSplitter运行流程图**
![[images/Pasted image 20260719103929.png]]
  
**衍生代码分割器**
递归字符文本分割器的核心部分在于传递不同的分割符列表，通过不同的优先级的列表，可以实现一些复杂文件的拆分，例如在该分割器内部预先构建了大量的的分割列表，用于在特定的编程语言中拆分文本。

支持的编程语言类型存储在 `langchain_text_splitters.Language`枚举中，涵盖常见编程语言

```python 
elif language == Language.PYTHON:
    return [
        # First, try to split along class definitions
        "\nclass ",
        "\ndef ",
        "\n\tdef",
        # Now split by the normal type of lines
        "\n\n",
        "\n",
        " ",
        "",
    ]

elif language == Language.RST:
```


**中文场景下的递归分割**

`RecursiveCharacterTextSplitter`默认配置的分隔符均是英文场合下的，在中文场合下，一般还有更加复杂的语句结束判断标识，例如：。、！、？等标识符，中文场景可以考虑重设分隔符列表（或者继承该类进行重写）。

不同符号的优先级如下：

1. `\n\n`：换行两次优先级最高
2. `\n`：普通换行符优先级其次
3. `。|！|？`：中文中句号、感叹号、问号一般都表示句子结束
4. `\.\s|\!\s|\?\s`：点、感叹号、问号，并且标准的英文写法在这些符号后通常需要添加空格。
5. `；|;\s`：其次就是中英文的分段，在英文分段后一般会添加空格；
6. `，|,\s`： 中英文中的逗号，逗号一般都表示句子语义还未结束，所以一般不切割，除非文本块仍然超过大小。
7. ：空格和空字符串是优先级最低的切割符号之一

#####  语义文档分割器与其他内容分割器

字符文档分割器都是使用特定字符对文本进行拆分，这种拆分模式虽然考虑了文档中的上下文打断的问题，但是并没有考虑句子之间的**语义相似性**，如果有一篇长文本，需要将其分割成语义相关的块，以便更好地理解和处理，这时可以用 LangChain 中语义相似性分割器（SemanticChunker）来实现

**需要传入embedding模型**，底层使用向量的余玹相似度识别予以之间的相似性

SemanticChunker 的原理：
将文本拆分成独立的每一句，接下来根据传递的缓冲大小前后拼接字符串，然后计算拼接后的新字符串的文本嵌入/向量，然后计算这些文本的相似度，并根据传入的分块数+断点类型计算得到一个阈值，最后将相似度超过某个阈值的合并到一起，从而实现相似度分割。

目前在 SemanticChunker 底层检测相似度阈值的方法有 4 种：百分位数（默认）、标准差、四分位数、梯度

![[images/Pasted image 20260719125121.png]]

**其他文档分割器**
除了文档分割器，LangChain 中还封装了一些其他分割器（使用频率不高），涵盖了：
* 基于 HTML 标题/段的分割器HTMLHeaderTextSplitter
* Markdown 标题分割器MarkdownHeaderTextSplitter
* 递归 JSON 分割器：按深度优先遍历json，进行分割，所以如果要使用该分割器，一般会结合 RecursiveCharacterTextSplitter 降低单条数据超过预设大小的风险，思路就是将递归 JSON 分割器生成的文档列表进行二次分割。
* 基于 Token 计数的分割器等：对于大语言模型来说，上下文的长度计算应该通过 token 进行计算，在 LLM 应用开发中，不同的模型对于 Token 的计算并不相同，但是可以使用 tiktoken 这个包来大致计算文本的 token 数，当你需要精确控制每个 chunk 不超过模型的 token 上限时（比如某些模型 context window 很小），按 token 切分比按字符切分更准确。但大多数时候字符切分已经够用了。

##### 自定义文档分割器

如果内置的文档分割器均没办法完成需求，还可以根据特定的需求实现自定义文档分割器（一般极少），实现的方法也非常简单，继承文本分割器基类 TextSplitter，在构造函数中传递相关参数，然后实现 split_text() 方法即可。

#### 6.1.3 文档转换器

在 LangChain 中，还存在另一种非分割类型的文档转换器，这类转换器也是传递 **文档列表**​ 并返回 **文档列表**，一般是将某种文档按照需求转换成另外一种格式（例如：**翻译文档、文档重排、HTML 转文本、文档元数据提取、文档转问答等**）。

这类文档转换器由于接收 **文档列表**，返回的也是 **文档列表**，所以可以在 LLM 应用中任何存在 **文档列表**​ 的地方使用，例如下方的 LLM 应用架构流程图中的 **文档加载**、**文档切制**、**检索器检索**​ 的环节交互数据都是 **文档列表**，所以这几个环节都可以添加文档转换器组件

![[images/Pasted image 20260719133632.png]]

##### 问答转换器

类：DoctranQATransformer，底层库doctran
- RAG优化价值：将叙述文本转为问答格式可提升检索相关性
    - 用户查询多为问题形式
    - 向量化QA对能更好匹配问题查询
- 微调应用：适合生成LLM微调所需的QA数据集

##### 翻译转换器

类：DoctranTextTranslator，底层库doctran
- 跨语言检索原理：语义相近的文本在向量空间位置接近
- 两种实现策略：
    - 入库前翻译多语言版本
    - 检索时实时翻译结果
### 6.4  向量数据库及检索器

目前市面上的向量数据库众多，每个操作方式也无统一标准，但是仍然存在着一些公共特征，LangChain 基于这些通用的特征封装了 vectorstore 基类，在这个基类下，可以将方法划分成 6 种：相似性搜索、最大边际相关性搜索、通用搜索、添加删除精确查找数据、检索器、创建数据库，类图如下：  
![[images/Pasted image 20260719134512.png]]

#### 6.4.1 VectorStore

##### 带得分阈值的相似性搜索
在 LangChain 的相似性搜索中，无论结果多不匹配，只要向量数据库中存在数据，一定会查找出相应的结果，在 RAG 应用开发中，一般是将高相似文档插入到 Prompt 中，所以可以考虑添加一个相似性得分阈值，超过该数值的部分才等同于有相似性。

在 similarity_search_with_relevance_scores() 函数中，可以传递 score_threshold 阈值参数，过滤低于该得分的文档。

对于 score_threshold 的具体数值，要看相似性搜索方法使用的逻辑、计算相似性得分的逻辑进行设置，并没有统一的标准，并且向量数据库的数据大小也存在间接关系，数据集越大，检索出来的准确度相比少量数据会更准确。

##### ⭐最大边际相关性搜索

MMR**核心原理：**
* 双重考量：最大边际相关性（MMR, max_marginal_relevance_search）的基本思想是同时考量查询与文档的 **相关度（R）**，以及文档之间的 **相似度（S）**。
	* 相关度 确保返回结果对查询高度相关
	* 相似度 则鼓励不同语义的文档被包含进结果。
	* 计算方式：`Score=λ⋅R−(1−λ)⋅S`，其中`λ﻿`为多样性系数
	* 筛选逻辑：在fetch_k条相似结果中，选择k条最不相似的文档

**实现过程：**
* 两阶段搜索：
	* 初步筛选：执行相似性搜索获取fetch_k条结果（默认20条）
    - 精筛：计算文档间余弦相似度，按MMR公式加权得分
- 参数设置：
	- lambda_mult：0表示最大多样性，1表示最小多样性
    - 典型设置：λ=0.5时，相关性与相似度各占50%权重
- 所以 MMR 在保证查询准确的同时，尽可能提供多样化结果，以增加信息检索的有效性和多样性

MMR 的运行演示图如下：

![[images/Pasted image 20260719140348.png]]

> Important
> 
> 简单来说，MMR 就是在一大堆最相似的文档中查找最不相似的，从而保证 结果多样化

**应用优势**

- 解决重复问题：当数据库存在相似文档（相似度>90%）时：
    - 传统搜索：可能返回多个相似结果
    - MMR搜索：自动剔除高相似文档
- 典型场景：
    - 用户重复上传相同文档
    - 文档存在微小差异（如仅差1-2个字）
- 价值：
	- 提升信息检索的有效性
	- 避免结果同质化
	- 适合需要创意组合的应用场景

**代码示例**

```python
search_docs = db.max_marginal_relevance_search(
    query="关于应用配置的接口有哪些？",
    k=4,                  # 返回结果数
    fetch_k=20,           # 初步筛选量
    lambda_mult=0.5       # 多样性系数
)
```
####  6.4.2  检索器
##### as_retriever()检索器
VectorStore 可以通过 as_retriever() 方法转换成检索器，在 as_retriever() 中可以传递一下参数：
1. search_type：搜索类型，支持 similarity(基础相似性搜索)、similarity_score_threshold(携带相似性得分+阈值判断的相似性搜索)、mmr(最大边际相关性搜索)。
2. search_kwargs：其他键值对搜索参数，类型为字典，参数的具体信息要看 search_type 类型对应的函数配合使用。

注：检索器是 Runnable 可运行组件，可以使用 Runnable 组件的所有功能（组件替换、参数配置、重试、回退、并行等）。

##### BaseRetriver检索器基类
![[images/Pasted image 20260719142612.png]]
在LangChain中接收query并返回相关文档的组件

##### VectorStoreRetriever检索器

VectorStoreRetriever 是 BaseRetriever 的子类，这是一个专门针对向量数据库的基础检索器，内部实现了 _get_relevant_documents() 方法，还定义了单独的属性：

1. vectorstore：检索器归属的向量数据库。
2. search_type：搜索类型
3. search_kwargs：搜索参数

这些参数均来源于 as_retriever() 或者在实例化类时传递的参数，由于该组件是一个 Runnable 可运行组件，所以可以使用 .configurable_fields() 来修改类内部的参数。

![[images/Pasted image 20260719143047.png]]
##### 内置检索器组件与自定义检索器

- LangChain内置了多种第三方检索器，包括维基百科搜索、Weaviate混合搜索、Zep检索器、ES搜索等国外服务，但有使用限制： 大部分检索器开发较早，可能不是Runnable可运行组件；使用方法存在差异，需查阅文档；主要面向国外产品，国内适用性有限
- 自定义检索器：继承BaseRetriever类，实现_get_relevant_documents方法在方法中完成从query到list[document]的转换逻辑
## 7. RAG优化策略

在RAG应用开发中，想进行优化，可针对query（提问查询）、Textsplitter（文本分割器）、vectorstore（向量数据库）、Retriever（检索器）、Prompt（基础prompt编写）这几个组件


### 7.1 ⭐查询转换阶段

#### 7.1.1 多查询重写策略提升检索准确性﻿

##### **多查询策略**

**原理：**
也称为“子查询“，核心思想：通过生成与主问题相关的子问题，从多个角度理解用户问题

**运行步骤：**
1. 将原始问题传递给大语言模型生成多个子问题（如3个）
2. 每个子问题分别执行检索
3. 合并所有检索结果并去重
4. 2将去重后的文档与原始问题一起传递给大语言模型生成最终答案
![[images/Pasted image 20260720220141.png|506]]

技术特点：
最简单的RAG优化策略之一（一般配合tempurature为0使用）；对LLM能力要求不高；适合本地小模型会增加单次对话耗时（多一次LLM调用）

适用场景：
用户问题较长或模糊时效果显著；可克服基于距离的相似性搜索限制

##### 多查询策略的检索器﻿ MultiQueryRetriever
- 关键参数：
    - retriever：基础检索器（必填）
    - llm：用于问题转换的大语言模型（必填）
    - prompt：问题转换模板（非必填，有默认值）
    - include_original：是否保留原始问题检索（默认False）
注：通过LangSmith平台可观察完整的执行流程

#### 7.1.2   多查询结构融合策略

多查询策略的核心问题：
* 文档数量失控：原始k=4时可能返回16个文档（3条子查询+1条原
* 权重缺失：仅按默认顺序合并，未考虑文档重要性

多查询结果融合：**在Multi-Query重写策略基础上，对其检索结果进行重新排序（即reranking）后输出Top K


流程优化：Q1→检索→合并筛选→RRF算法→LLM生成答案
![[images/Pasted image 20260720221408.png]]

#####  RRF

Reciprocal Rank Fusion 倒排序排名算法

![[images/Pasted image 20260720221521.png]]
        
D：相关文档全集
k：固定常数60（经实验验证的最优值）
r(d)：文档d在子集中的排名位置
    
> 排名特性：虽然高排名文档更重要，但低排名文档权重不会指数级衰减

##### 自定义检索器实现融合

langchain里没有可以直接用的类，需要继承MultiQueryRetriever类自定义实现

#### 7.1.3 问题分解策略提升复杂问题检索正确率
##### 复杂问题检索的难点与分解

**复杂问题检索的难点**
- 检索效果偏差现象：对于复杂的原始问题，无论是直接检索还是生成关联问题检索，都难以在向量数据库中找到高关联文档。
- 典型案例：如机器说明文档中查询"如何完成某个部件维修"这类多步骤问题，相似性搜索往往无法找到关联文档。
- 根本原因：
    - 文本嵌入模型限制：复杂问题由多个顺序步骤组成，但向量数据库存储的是基础文档数据，相似度低；且单条向量无法无损记录完整段落信息。
    - 上下文长度限制：高复杂度或数学问题导致LLM无法一次性生成答案，大量相关文档会压缩模型上下文窗口

##### 问题分解策略
﻿
将复杂问题分解为多个子问题，与多查询重写策略的关键区别在于采用深度优先处理方式。

两种实现方案：
1. 迭代式回答：解决完第一个问题后，将答案传递给第二个问题，依次类推

![[images/Pasted image 20260721212605.png]]
2. 并行式回答：并行处理所有子问题，最后合并答案
![[images/Pasted image 20260721212743.png]]
#### 7.1.4 Step Back回退策略扩大检索范围
##### Few Shot

在于LLM的对话中，提供少量的示例被称为少量实例（与之对应的是零样本）

langchain中的提示模板：`FewShotPromptTemplate(example_prompt. examples)`

##### Step-Back

对于一些复杂的问题，除了使用 问题分解 来得到子问题亦或者依赖问题，还可以为复杂问题生成一个前置问题，通过前置问题来执行相应的检索，这就是 setp-Back 回答回退策略（后退提示）。这是一种用于增强语言模型的推理和问题解决能力的技巧，它鼓励 LLM 从一个给定的问题或问题后退一步，提出一个更抽象、更高级的问题，涵盖原始查询的本质。

Step-Back 回答回退策略的运行流程：
构建一个 少量示例提示模板，让 LLM 根据传递的问题生成一个后退问题，使用 后退问题 执行相应的检索，利用检索到的文档+原始问题执行 RAG 索引增强生成，运行流程如下：
![[images/Pasted image 20260721215001.png]]

#### 7.1.5 混合策略实现doc-doc对称检索

##### HyDE混合策略

传统查询优化策略存在query和doc不对称检索问题，数据库存储的是文档级数据，直接搜索用户原始查询可能因无关信息干扰导致检索效果不佳。

HyDE混合策略思想：通过LLM将问题转换为**回答问题的假设性文档/假回答**，然后用**嵌入的假设性文档去检索真实文档**，前提是doc-doc这个模式执行相似性搜索可以尝试更多的匹配项目。

> 假回答和真回答可能在事实上存在错误，但语义空间更接近。

![[images/Pasted image 20260721215810.png]]

**局限性与失败案例

本质局限：HyDE是一个无监督方法，效果依赖于LLM对问题的理解准确性，误解会导致错误累积

#### 7.1.6 集成多种检索器算法实现混合检索

核心原理: 通过**EnsembleRetriever**封装多个检索器，使用RRF算法对结果进行集成和重新排序

互补优势: 例如稀疏检索器(如BM25)擅长关键词检索；密集检索器(如嵌入相似度)擅长语义相似性检索

应用场景: **广泛应用**于Dify、coze、智谱等AI平台

#### 7.1.7 总结

在 RAG 的 查询转换 阶段的主流优化策略：多查询重写、RAG 多查询结果融合、问题分解策略、回答回退策略、HyDE 混合策略、集成检索器策略等，不同的优化策略有不同的优缺点：


###### 1）**多查询重写**
- 优点:
    - 实现简单，可使用小参数模型
    - 支持并行检索，性能较高
- 缺点:
    - 合并时未考虑文档权重
    - 可能剔除高权重文档

###### 2）多查询融合
- 改进点: 引入RRF算法计算文档权重
- 排序规则: 将高频出现或排名靠前的文档置于合并结果前列
###### 3）问题分解策略
- 流程:
    - 将复杂问题分解为子问题
    - 对每个子问题独立检索-生成
    - 合并子问题答案
- 局限:
    - 对LLM要求高
    - 上下文不足时可能导致答案偏离
###### 4）回答回退策略
- 原理: 通过前置问题扩展搜索范围
- 优势:
    - 不涉及中间回答，可使用小模型
    - 检索性能较高
###### 5）HyDE混合策略
- 核心思想: 将query转换为doc实现对称检索
- 局限: 对开放性问题效果较差

###### 6）集成检索器策略
- 当前地位: 使用频率最高的检索策略
- 关键技术:
    - 融合多种检索算法优势
    - 采用RRF算法合并结果

### 7.2 路由阶段
#### 7.2.1 检索器逻辑路由

**函数回调**

**大模型函数回调的理解**﻿
- 核心概念：通过向大语言模型传递工具/函数信息（包括函数名、参数描述、功能说明），让模型自主判断在当前用户提问下最适合调用的函数
- 输出特点：不同于常规字符串输出，函数回调模式下模型会返回JSON格式的函数调用参数
- 执行机制：模型仅生成函数调用参数，实际函数执行由本地程序完成
- 技术优势：使模型具备智能工具选择能力，同时实现输出内容规范化

**函数回调的运行流程**

![[images/Pasted image 20260722214219.png]]

**函数回调的调用模式**
- 自动模式(auto)：模型自主决定是否调用函数（有可用函数时的默认模式
- 禁止模式(none)：强制模型不调用任何函数（无可用函数时的默认模式
- 强制模式(required)：强制模型必须调用指定函数（需配合具体函数定义使用
- 模式选择：通过tool_choice参数设置（OpenAI API）


**利用函数回调规范化输出**﻿

- 原理：通过构建假函数并强制LLM调用，将需要规范化的数据写成函数参数并配上解释，实现比Prompt更可靠的输出约束
- 实现方法：使用LangChain的with_structured_output()方法，传递BaseModel子类，底层会自动转换为函数回调

**检索器的逻辑路由实现**﻿

**设定对应的 Prompt，然后让 LLM 根据传递的问题返回需要选择的 检索器 的名称，然后根据得到的名称选择不同的检索器即可。**

注：对于 LLM 来说，如果使用普通的 prompt 来约束输出内容的格式与规范，因为 LLM 的特性，很难保证输出格式符合特定的需求，所以可以考虑使用 `函数回调`来实现，即设定一个 虚假的函数，告诉 LLM，这个函数有对应的参数，让 LLM 强制调用这个函数，这个时候 LLM 就会输出函数的调用参数，从而保证输出的统一性。

使用函数回调实现的检索器逻辑路由：
![[images/Pasted image 20260722215739.png]]

#### 7.2.2 提示模板路由

> **针对不同场景的问题使用特定化的prompt模板**效果优于通用模板

所以利用向量执行相似性搜索，不仅可以作用于 向量数据库，我们还可以利用 **原始问题 与 prompt模板 的相似性，来找到类型、语义上更接近的模板**，从而实现对 prompt模板 的动态路由。

![[images/Pasted image 20260722221102.png]]

###  7.2  上下文构建阶段
#### MultiVector实现多向量检索文档

#### 7.1.8 子查询检索器实现动态元数据过滤


#### 父文档检索器

#### 递归文档检索树

### 7.3 重排序阶段

#### ReRank

### 7.4 生成阶段

### 7.4.1 纠正性索引增强生成CRAG
#### 7.4.2 self-RAG 纠正低质量的检索生成