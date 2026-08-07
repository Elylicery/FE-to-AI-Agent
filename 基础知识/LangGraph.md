
## LECL表达式的局限性

有向无环图DAG（Directed Acyclic Graph）结构是LCEL表达式构建链应用的基础框架

LCEL的局限性：
* 线性流程缺陷：只能按定义顺序执行、无法实现动态条件分支
* 状态管理缺失：需在每次链调用时显示传递记忆信息
* 工具集成缺陷：动态选择工具困难；工具层级增多时构建难度指数级上升


## LangGraph介绍

LangGraph核心优势

* **循环机制** ：实现循环和条件语句
* **可控制性**：在图的每个步骤之后自动保存状态，在运行的任意一个阶段都支持暂停和恢复图执行，以支持错误恢复、人机交互工作流、时间旅行等
* **人机交互**：中断图的执行以批准或者编辑状态计划去执行下一计划
* **流支持**：节点支持流式输出

**langchian与langGraph**
- LangChain：
    - 工具库性质
    - 线性流程为主
    - 适合简单任务组合
- LangGraph：
    - 流程引擎性质
    - 支持状态机/循环/分支
    - 适合复杂业务流程

## LangGraph基础组件

#### 关键组件

三个关键组件来定义Agent行为:
![[images/Pasted image 20260718165957.png]]

1. 状态（State Attribute）：状态是图中所有 节点 和 边 的输入，它可以是一个 dict(字典) 或者 Pydantic 模型，状态包括 图的模式(数据结构) 以及如何更新状态的 归纳函数，如果没有设置 归纳函数，则每次节点都会覆盖 状态的原始数据。
2. 节点（Node）：节点通常是 Python 函数（同步或异步），其中节点函数的第一个参数是 state(状态)，第二个参数是 config(Runnable 运行的配置)，节点函数的返回值一般都是 状态
3. 边（Edge）：边在图中定义了路由逻辑，即不同节点之间是如何传递的，传递给谁，以及图节点从哪里开始，从哪里结束，并且一个节点可以设置多条边，如果有多条边，则下一条边连接的所有节点都会并行运行。

在 LangGraph 中，无论多么复杂的项目，都可以拆分成对应的 6 个步骤，只需按照这 6 个步骤来执行即可：
1. 初始化大语言模型和工具（ChatOpenAI、tools）
2. 用状态初始化图架构（StateGraph 状态图）
3. 为图定义每一个节点（add_node 函数为图添加节点）
4. 定义图的起点、终点和节点边（add_edge 函数为图添加边）
5. 编译图架构为 Runnable 可运行组件（graph.compile 函数编译图）
6. 调用编译后的 Runnable 可运行组件执行图（graph.invoke 函数调用图）
 
## LangGraph实现ReACT架构
### 条件边与循环流程

普通边 add_edge()
条件边 add_conditional_edge()

> 对于循环流程，一般都会有一个条件边用于跳出循环，都则LangGraph会检测到没有跳出循环的条件，应用程序会崩溃且消耗巨大系统资源

### create_agent 构建器

在 LangGraph 中除了可以搭节点、变来构建 Agent 智能体，这也是 LangGraph 自由性高的一个优点，我们还可以使用预构建函数（实际是语法糖）`langchain.agents.create_agent` 构建智能体
### 其他预构建组件

langgraph内置了一些高频使用的预构建组件，例如
* `MessageState` 自定义状态
* `toolNode` 节点就是工具执行节点，该节点会自动执行数据状态中最后一条消息中的工具调用信息
* `validationNode` 检验最后一个AiMessage中所有工具请求是否正确，用于校验LLM的结构化输出是否正确
* `InhectedState` 用于在工具上注入数据状态，这样在工具中也可以获取到`图架构应用`的`数据状态` 
## 图结构应用程序中删除消息

### 更新删除与归纳函数

使用RemoveMessage装饰器配合add_messages()函数

> Important
> 
> 删除消息一定要特别注意，因为绝大部分模型期望消息列表存在某些规则。例如，有些模型期望它们以 user 消息开头，其他模型期望所有带有工具调用的消息后面都跟着工具消息。**删除消息时，需要确保不会违反这些规则。**
### 过滤与修剪消息

在 LangGraph 中 状态 可以很便捷管理整个过程中产生的所有消息信息，但是随着持续对话，亦或者图结构组件的增加，对话历史会不断累积，并占用越来越多的上下文窗口，这通常是不可取的，因为它会导致对 LLM 的调用变得非常昂贵和耗时，并降低 LLM 生成内容的正确性，所以在 LangGraph 中一般还需要**对消息进行过滤和修剪**

**过滤/修剪 一般不会更改 状态，而是在调用 LLM 时，只传递特定条数的消息或者按照 token长度 进行修剪。**

langchain对于该需求封装了特定函数 trim_messages

## 检查点实现记忆持久化功能

**检查点与线程**

在 LangGraph 中，持久化使用的是 **检查点** ，并每个 检查点 还和 线程ID 有关， **检查点** 存储的并不是整个图结构应用程序的 节点状态 ，而是**存储 特定线程 的数据状态**，这是因为 LangGraph 在设计的时候就考虑到一个应用的多次独立对话功能。


**检查点实现记忆持久化功能**

 检查点 来为图提供持久化记忆，操作两步：
1. 实例化一个检查点，例如 `AsyncSqliteSaver`或者 `MemorySaver()`，亦或者自定义检查点。
2. 在图编译的时候传递检查点，例如 `compile(checkpointer=my_checkpointer)`
接下来在和图程序交互时传递`config`，并配置`thread_id` 即可记住以往的历史记忆/存档

```python
# 使用预构建create_react_agent并传递检查点
checkpointer = MemorySaver()
agent = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

# 调用智能体并输出内容
print(agent.invoke(
    {"messages": [("human", "你好，我叫小课，我喜欢游泳打球，你喜欢什么呢？")]},
    config={"configurable": {"thread_id": 1}}
))

# 二次调用
print(agent.invoke(
    {"messages": [("human", "你知道我叫什么吗？")]},
    config={"configurable": {"thread_id": 1}}
))

```

**LangGraph其他检查点**

在 LangGraph 中，除了封装了 MemorySaver 基于 临时内存 的检查点，还封装了基于 Postgres、MongoDB 和 Redis 的检查点。

这些检查点的运行流程都一模一样，只是持久化/存储的介质不一样而已，根据存储方式的不同使用不同的实例化方式。

## 图结构断点实现Agent与人进行交互

### 人在环路与断点

**人在环路（Human-in-the-loop，简称 HIL）** 交互对于 Agent 系统至关重要，特别是在一些特定领域的 Agent 中，需要经过人类的允许或者指示才能进入下一步（例如某些敏感或者重要操作），而 HIL 最重要的部分就是 断点。

断点 建立在 LangGraph 检查点之上，检查点在每个节点执行后保存图的状态，并且 检查点 可以使得图执行可以特定点暂停，等待人为批准，然后从最后一个检查点恢复执行。

流程图

![[images/Pasted image 20260806211053.png]]

```python
# 编译图为Runnable可运行组件
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])

# 调用图架构应用
config = {"configurable": {"thread_id": 1}}
state = graph.invoke(
    {"messages": [("human", "2024年北京半程马拉松的前3名成绩是多少")]},
    config=config,
)
print(state)

# 获取人类提示
if hasattr(state["messages"][-1], "tool_calls") and len(state["messages"][-1].tool_calls) > 0:
    tool_calls = state["messages"][-1].tool_calls
    print("准备调用工具：", tool_calls)
    human_input = input("如需调用工具请回复yes，否则回复no：")
    if human_input.lower() == "yes":
        print(graph.invoke(None, config))
    else:
        print("图程序执行结束")


```

### 在图结构上更新对应状态
在图结构上，除了能通过 节点 更新 数据状态，还可以在图的外部通过调用图的 `get_state()` 与` update_state()` 的方式来实现对数据状态的更新（特定线程下），并且 `get_state()` 和 `update_state()` 功能必须在 检查点 模式下才支持。

```python
# 6.更新图的状态，去篡改工具消息
graph_state = graph.get_state(config)
tool_message = ToolMessage(
    # id是告诉归纳函数我和原始数据重复了，请直接覆盖
    id=graph_state[0]["messages"][-1].id,
    # 告诉大语言模型工具调用id，这里的工具调用id是让大语言模型知道这条消息是和哪个函数关联
    tool_call_id=graph_state[0]["messages"][-2].tool_calls[0]["id"],
    name=graph_state[0]["messages"][-2].tool_calls
```

## 子图架构实现AI工作流

对于一些更加复杂的系统，子图是一个非常有用的设计原则。使用子图允许在图的不同部分创建和管理不同的状态，这样可以轻松利用 LangGraph 实现类似 多智能体 亦或者 AI工作流 之类的功能，每个功能之间相互独立隔离，最后组装成一个大型复杂应用。

创建 子图 最核心的部分要认识到 图 之间的信息传递，入口图 是父级，两个子图都被定义成 入口图 的节点，并且两个子图都继承了父级 入口图 的状态，并且每个子图都可以拥有自己的私有状态，任何想传回父级 入口图 的值，只需要在入口图中定义即可。

![[images/Pasted image 20260806214416.png]]

## LangGraph实现CRAG图

 CRAG（纠正性索引增强生成）优化策略，在该优化策略中引入了一个轻量级的评估器用于评估检索到的文档的质量，并根据评估结果触发不同的知识检索动作，
 
 运行流程：
![[images/Pasted image 20260807151113.png]]

## 两种基础流式响应

LangGraph 中，流式响应每次输出的都是一个节点的数据状态，流式模式有两种：
- `values`：此流式模式返回图的值，这是每个节点调用后图的`完整状态`（总量）；
- `updates`：此流式模式返回图的更新，这是每个节点调用后图的`状态更新`（增量）；

使用：调用 stream() 函数时，传递 stream_mode 参数配置不同的流式响应模式



## 其他输入（补充进文档里）
##### 多Agent系统实现
- 典型架构：
    - 客服Agent：接收用户请求
    - 订单Agent：查询订单详情
    - 物流Agent：跟踪运输状态
- 协作机制：
    - 路由用户问题到对应Agent
    - 工具调用结果聚合
    - 统一响应生成


##### 性能优化策略
》
- 并行化：
    - 使用Worker-Aggregator模式
    - 并发执行独立子任务
- 流程优化：
    - 将串行步骤拆分为可并行单元
    - 关键路径上使用轻量级模型
- 缓存策略：
    - 缓存频繁查询结果
    - 注意敏感数据隔离

##### 6. 错误处理机制

- 防御性设计：
    - 参数校验前置（防SQL注入）
    - 设置超时熔断机制
- 恢复策略：
    - 自动重试（指数退避）
    - 备用工具降级方案
- 监控体系：
    - 记录完整调用链日志
    - 关键指标实时告警