## LLM

LLM全称Large Language Model，即大语言模型，是一种给模型一些输入，它可以用大量数据训练的深度学习模型预测并返回相应的输出

本质：连续预测下一个Token的生成系统

<details >

<summary>📝 AIGC与AGI是什么</summary>

AI-Genrate Content，通过对已有数据进行学习和模式识别，**以适当的泛化能力生成相关内容的技术**

**AGI（artifical general intelligence）全称人工通用智能**，是指能够理解、学习和应用广泛的知识和技能的人工智能系统
</details>

### Token

**大语言模型的基础单位Token**

- Token其实就是文本的片段，是大模型**计算长度**的单位，对于汉字，可以是字、词、甚至是半个字或者三分之一个字。
- 对于一个**仅支持英语**的模型，它的词表可以只有a-z26个字母，加上逗号句号空格等标点符号，Token数可以非常少但是一般情况下都会添加单词作为最基础单位)。
- 汉字字词更多，语义更复杂多样，所以包含的 Token 数会更多，很多语言模型都支持多语言，包含各种符号、单词、单词片段等，所以往往会有几十万个 Token 甚至更多。


**大语言模型词表**
词表，就是这个模型的 ，每个 Token 有其对 应的 id，一般从 0 开始，如下

```json
{
  "vocab": {
    // 开头是一些特殊符号
    "<unk>": 0,
    "<|startoftext|>": 1,

    // 这是字节token，如果出现不在词表中的特殊符号会回退到字节表示
    "<0x00>": 305,
    "<0x01>": 306,
    "...": null,

    // 下面是正常的英文token，有_的表示是单词的开头，没有的是单词中间
    "ct": 611,
    "_re": 612,
    "...": null,

    // 有中文token出现
    "安徽省": 28560,
    "子和": 28561,
    "...": null
  }
}
```

**大模型预测的工作流程**
![[images/Pasted image 20260627210308.png]]
### 常见专有名词

LoRA-插件式微调

- LoRA（Low-Rank Adaptation of LLM）即**插件式微调**，用于对大语言模型进行个性化的特定任务的定制。
- LoRA通过将模型的权重矩阵分解成低秩的相似矩阵，降低了参数空间的复杂性，从而减少微调的计算成本和模型存储要求

矢量/向量数据库

- 矢量数据库是一种用于存储 **矢量/向量数据**的数据库。
- 矢量数据库可以存储和管理大量的矢量数据，例如图像、视频、音频、文本等，同时提供高效检索功能。

数据蒸馏

- 数据蒸馏指将给定的原始大数据集浓缩并生成一个小型数据，使得在小数据集上训练出来的模型与原数据集上训练的模型相似。
- 数据蒸馏在深度学习领域被广泛应用，可以帮助将复杂的模型转换成更轻量级的模型，提高模型的鲁棒性和泛化能力。

### 结合大模型的新一代应用交互方式

- **嵌入（Embedding）模式**：用户通过与AI交流，AI协助完成，如创作小说、音乐、3D内容等，此模式下，AI是执行工具，人类是决策者和指挥者。
- **副驾驶（Copilot）模式**：在这种模式下，人类与AI作为合作伙伴，共同完成任务，AI提供建议并协助任务，二者互补，AI更像知识丰富的伙伴而非工具。
- **智能体（Agent）模式**：人类设定目标并提供资源，AI独立完成大部分工作，最后人类监督和评估结果。

## Prompt Engineering
### 意义

**目的**：提升AI推理质量与可控性，精细化控制模型的推理质量、行为边界和输出风格。
比如：设计Tool Calling Prompt→统一接口规范，降低维护成本

Prompt不只是改变大模型的输出内容与格式，也是驱动Agent的最重要手段。
比如：意图识别、任务分解、外部工具调用等关键功能。
### Prompt工程

#### **推理增强技术**
* 思维链（Chain-of-Thought）：
	* “Let’s think step by step”
	* 示例：数学题、逻辑推理
* 反思机制(Self-Reflection)：模型自我评估与修正
	* 模型自我评估输出质量
	* 使用Critic Agent或prompt设计

#### **提示模板工程**

* LangChain PromptTemplate / Jinja2
```python
from langchain_core.prompts import PromptTemplate

# 使用 Jinja2 语法，支持 if/else
template = """
你是一个客服助手，请根据用户问题提供帮助。

用户问题: {{ query }}

{% if product_info %}
产品信息: {{ product_info }}
请基于以上信息回答。
{% else %}
该产品暂无详细信息。
{% endif %}

回答:
"""

# 创建支持 Jinja2 的 PromptTemplate
prompt = PromptTemplate.from_template(template, template_format="jinja2")

# 格式化输出 (带条件)
formatted_prompt = prompt.invoke({
    "query": "iPhone 15 有货吗？",
    "product_info": "iPhone 15 256GB 版本有现货，售价 6999 元。"
})

print(formatted_prompt
```
* 支持变量注入、条件判断
```python
template = """
你是一个旅游推荐官。

目的地: {{ city }}
天数: {{ days or 3 }}
预算等级: {{ budget_level | default('中等', true) }}

请推荐一个 {{ days }} 天的行程，预算 {{ budget_level }}。
"""

prompt = PromptTemplate.from_template(template, template_format="jinja2")

# 测试：只传 city，其他用默认值
formatted = prompt.invoke({"city": "京都"})
print(formatted.text)
```
* 动态Prompt生成（函数化构造）
```python
def create_dynamic_prompt(intent: str):
    templates = {
        "refund": """
你是一名售后客服。
用户申请退款，订单号: {{ order_id }}，原因: {{ reason }}。
请礼貌回复并说明处理流程。
""",
        "support": """
你是一名技术支持。
用户遇到问题: {{ issue }}。
环境: {{ environment }}。
请提供排查建议。
""",
        "recommend": """
你是一名推荐助手。
用户偏好: {{ preference }}。
请推荐一个 {{ preference }} 风格的产品。
"""
    }
    return PromptTemplate.from_template(templates.get(intent, "请回答: {{ input }}"))

# 根据用户意图动态选择模板
user_intent = "refund"
prompt = create_dynamic_prompt(user_intent)

input_data = {
    "order_id": "ORD123456",
    "reason": "商品发错"
}

formatted = prompt.invoke(input_data)
print(formatted.text)
```

* 意图识别
```python
# 步骤 1: 用 LLM 判断用户意图
intent_prompt = PromptTemplate.from_template(
    "判断用户问题的意图，仅返回一个关键词: refund, support, recommend, general\n"
    "用户问题: {{ query }}\n"
    "意图: "
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 完整链：用户输入 -> 判断意图 -> 动态生成 Prompt -> 最终回答
def build_smart_agent_chain():
    # 意图识别链
    intent_chain = intent_prompt | llm | StrOutputParser()

    def route_prompt(data):
        query = data["query"]
        intent = data["intent"]

        # 动态选择并格式化 Prompt
        dynamic_prompt = create_dynamic_prompt(intent)
        return dynamic_prompt.invoke({"input": query, **data})

    # 主链：先识别意图，再生成回答
    full_chain = (
        {"query": lambda x: x["query"]}
        | intent_chain.assign(intent=lambda x: x)  # 保留原始输入
        | route_prompt
        | llm
        | StrOutputParser()
    )
```
#### 外部工具调用提示设计：

（Tool Calling Prompt）
* *明确指令引导模型选择正确工具（如：“若需查询订单，请调用query_order_tool”）
* 提供清晰参数描述与示例，减少误调用

#### Few-shot Prompting与Example Selector

- **Few-shot Prompting**：用“示范”教模型怎么做
- **Example Selector**：从大量示范里，自动挑“最合适的那几个”
#### **安全与管理**

* Guardrails / Safe Prompting：防御提示注入、防止越权
* Prompt版本管理与A/B测试：科学迭代提示，评估效果
## Agent

OpenAI将AI-Agent定义: 以大语言模型为大脑驱动，具备**自主感知、规划、记忆和使用工具**能力，能自动执行复杂任务的系统

### Agent定义

智能体**感知环境**，并进行**自主思考**，**拆解任务并采取行动**完成一系列目标，这就是**Agent**。

![[images/Pasted image 20260627210400.png]]

开发Agent就是开发“控制循环”这部分

### Agent/LLM在企业有哪些应用场景落地?

- 体验类：办公助理、知识库助手
- 数据分析：企业每天产生的大量新数据和已有的海量历史数据（日志、历史代码、订单、会员信息、行业数据等），利用LLM自动分类提取，构建知识库问答机器人，快速完成历史数据传承

- 降本类：智能客服、运维管理

- 智能客服：利用LLM+企业知识库搭建AI智能客服，实现智能沟通、无间断工作、智能分析、自动分类、跨语言沟通
- 人工智能即服务-AIaaS（AI as a Service）：将AI视为应用后端服务，让公司发布一款产品变得及其简单，无需投入昂贵的硬件、专业人才或耗时的开发流程，例如翻译网站、自动化运维（LLM分析日志+调工具包

- 增收类：销售赋能、原生应用集成
- 变革类：流程智能化、实时预测
### Agent技术架构

![[images/Pasted image 20260627210628.png]]

Agent更聪明的四大核心特征：
- **自主性**（不用你教我做事）
- **反应性**（眼观六路、耳听八方）
- **目标驱动型**（想你所想，想你未想）
- **学习能力**（在经验中成长）
### 控制循环: Agent核心设计模式

- 控制循环也是Agent开发中最核心的部分，也是开发者编写的**Agent设计模式**，这个设计模式就是Agent的生命力所在。一个好的设计模式能让Agent变得更强大。
- 市面上相对成熟的Agent设计模式有5种：**反思模式、工具使用模式、ReACT模式、规划模式、多智能体协作模式**，不同模式的Agent有不同的适用场景。

![[images/Pasted image 20260627210828.png]]

![[images/Pasted image 20260627210844.png]]

![[images/Pasted image 20260627210856.png]]

![[images/Pasted image 20260627210907.png]]

![[images/Pasted image 20260627210923.png]]
### Agent进化史
Agent是怎么一步步变聪明的?

进化形态一：专家系统（1970s-1990s）

- Agent的初代形态是一个**一丝不苟的图书管理员**，其核心思想就是把人类专家的知识，变成一条条明确的IF...THEN...规则录入电脑。
- 专家系统获得了人类赋予的**领域知识(规则)**，在特定领域非常成功，比如MYCIN领域，用于诊断血液感染疾病，其准确率甚至超过了一些人类医生。
- 但其弱点也非常致命，系统**太脆弱**，规则之外的世界完全不懂；规则越多越乱，牵一发而动全身，**维护噩梦**；永远无法从新案例中自我进化，**无法自我学习**。

进化形态二：强化学习（2010s）

- 不直接教它知识，而是给它一个目标，让它自己在环境中摸爬打滚，**自己学习**，通过奖励惩罚，在尝试中学会最优策略，这就是**强化学习**。
- 强化学习可以让模型获得**强大的适应与学习能力**，最经典案例就是AlphaGo，它通过自我博弈，击败了世界顶级的围棋选手，它下的很多棋步，是人类几千年从未有的神之一手。
- 强化学习没造出通用管家，其缺陷也很明显，需要明确的奖励，但很多现实任务**无法量化奖励**；依赖海量模拟，现实世界**试错成本太高**；缺乏常识理解，**不懂语言**，不懂世界。

```
   -----------做出动作Action -------
  |                                ↓
agent                         Envirement 
  ↑                                |
  ----返回状态&奖励（state&Reward）-

```

最终形态：理解与推理的王者（2020s-）

- Agent的现在将LLM作为超级大脑指挥官，其核心思想是**以LLM作为通用世界模型和推理引擎**，
- LLM可以让Agent**理解世界知识与常识**，拥有强大的自然语言理解能力；拥有超强的**逻辑推理与规划**，能把一个大目标分解成可执行小步骤。

## MCP


### Function Call

![[images/Pasted image 20260627211754.png]]

#### 定义

OpenAI的Function Calling（现称为Tool Calling）是构建智能Agent的核心技术之一，它改变了传统LLM直接生成答案的
模式，转而让模型“思考”是否需要调用外部工具来完成任务。

一个Agent Loop（代理循环）：
User Input->
1. 向模型发起一个包含其可调用工具的请求
2. 接收来自模型的工具调用
3. 在应用程序端执行代码，使用工具调用的输入
4. 使用工具输出向模型发起第二次请求
5. 接收来自模型的最终响应（或更多的工具调用）

#### 工程意义

Function Call的工程意义

| **<br><br>维度<br><br>** | **<br><br>价值<br><br>**         |
| ---------------------- | ------------------------------ |
| 解耦                     | 模型只负责“要不要查”，不负责“怎么查”           |
| 智能调度                   | 实现多工具协同、条件判断、链式调用              |
| 易于扩展                   | 新增工具无需改主逻辑，只需注册                |
| 可观察性                   | 能追踪每一步决策（reasoning）和动作（action） |
| 安全性                    | 可在执行前审查 tool_call 参数（防注入）      |
| 支持复杂流程                 | 如：先查库存 → 再算价格 → 最后下单           |
|                        |                                |

**工程化进阶方向**
1. 工具注册中心：统一管理所有tools，支持动态加载
2. 参数校验中间件：防止恶意或错误参数传入
3. 超时与重试机制：工具调用失败自动重试
4. 缓存机制：相同参数避免重复调用
5. 权限控制：某些工具仅限特定角色调用
6. 日志追踪：记录完整的Agent Loop流程
7. 可视化调试面板：查看模型决策链路

### MCP

#### 定义

**模型上下文协议（MCP）** 是 统一大型语言模型（LLM）与外部数据源和工具之间的一种**通信开放协议**。**MCP 使得 AI 应用能够安全地访问和操作本地及远程数据，为 AI 应用提供了连接万物的接口。

**核心能力**
MCP是一种开放协议，它标准化了应用程序如何向LLM提供工具和上下文。

#### 两者关系区别

**关系**
* **MCP让FunctionCall变得标准化** ：FunctionCall借助MCP，不用客户端重复造轮子，实现函数定义与函数调用
* 统一接口：**MCP统一调用规则，通过list_tools提供标准化的函数列表给到大模型**，无论模型是否支持FunctionCall 
* 双模式支持： 
	* 支持FunctionCall的模型：将函数列表填入tools字段
	* 不支持FunctionCall的模型：通过提示词传递函数信息

**核心定位差异**
* Tool Calling：你告诉模型“你能用哪些工具”
* MCP：模型自动发现并使用互联网上的“标准化工具

| **<br><br>维度<br><br>** | **<br><br>Function / Tool Calling<br><br>** | **<br><br>MCP (Model Context Protocol)<br><br>** |
| ---------------------- | ------------------------------------------- | ------------------------------------------------ |
| 控制权归属                  | 开发者完全掌控工具定义与执行                              | 模型可调用第三方远程服务（非你部署）                               |
| 工具部署位置                 | 工具在你的代码中（本地或内网）                             | 工具在远程服务器上（如 mcp.stripe.com）                      |
| 信任模型                   | 完全可信（你自己写的）                                 | 需审慎信任第三方（可能恶意）                                   |
| 标准化协议                  | OpenAI 私有实现（非开放标准）                          | 开放协议（跨平台、多厂商支持）                                  |
| 典型用途                   | 调用内部 API、数据库、计算函数                           | 集成 SaaS 服务（Stripe、Shopify、Twilio等）               |

**使用场景差异**

| **<br><br>场景<br><br>**    | **<br><br>推荐方案<br><br>** | **<br><br>理由<br><br>** |
| ------------------------- | ------------------------ | ---------------------- |
| 调用内部数据库、私有 API            | Tool Calling             | 安全、可控、低延迟              |
| 快速接入 Stripe/PayPal/Twilio | MCP                      | 无需开发，开箱即用              |
| 多个 SaaS 系统联动              | MCP                      | 统一协议，避免重复对接            |
| 对安全性要求极高（金融、医疗）           | MCP + Approval           | 必须审查每个调用               |
| 希望构建通用 AI 工具平台            | MCP                      | 支持插件化、可发现、可组合          |
|                           |                          |                        |
|                           |                          |                        |
**Tool Calling之外的工程强化手段**
1. 流式输出（Streaming）
2. 异步调用与批处理（Async & Batching）
3. 超时控制与重试策略
4. 上下文管理与长度优化
5. 工具调用审批机制（Approval）
6. 结构化输出（JSON Schema）
7. 认证与安全头管理（Authentication）
8. 限流、熔断与降级策略
9. 监控、日志与链路追踪
10. 输入净化与提示词注入防护
### **MCP工作原理**

MCP 采用的是经典的客户端 - 服务器架构，几个概念:

- MCP 主机（MCP Hosts）：发起请求的 LLM 应用程序（例如 Claude Desktop、IDE 或 AI 工具）
- MCP 客户端（MCP Clients）：在主机程序内部，与 MCP Server 保持 1:1 的连接。
- MCP 服务器（MCP Servers）：为 MCP Client 提供上下文、工具和 prompt 信息。
- 本地资源（Local Resources）：本地计算机中可供 MCP Server 安全访问的资源（例如文件、数据库）。
- 远程资源（Remote Resources）：MCP Server 可以连接到的远程资源（例如通过 API）。

![[images/Pasted image 20260627211735.png]]

**MCP客户端作用是什么？**

- MCP分为MCP Server和MCP Client两部分
- MCP Server: 目前已有多种现成的MCP Server可直接对接，无需自行开发代码
- MCP Client特点: 只有一种，被集成到各类客户端中，用于与MCP server交互

### 通信方式与消息格式
#### 通信方式

![[images/Pasted image 20260627211907.png]]

- 客户端形式：VS Code插件（ClineCherryStudio）、本地模型客户端、编程工具（Cursor）、原生MCP Client开发
- 通信协议：stdio：进程间通过标准输入输出通信；SSE：基于HTTP长连接的服务器推送技术
- 服务端功能：本地文件操作、网络爬虫、外部API调用（天气/地图等）
- 执行阶段

- 客户端枚举可用函数列表
- 模型决定需要调用的函数
- MCP Server执行具体操作
- 返回结果给模型进行分析

#### 消息格式-JSON RPC2.0

请求示例
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "sum",
  "params": [5, 10]
}
```

-请求规范：
- 必须包含jsonrpc: "2.0"版本标识
- 需要唯一id实现请求-响应映射
- method指定调用函数名（如"sum"）
- params传递参数数组（如[5, 10]）

响应示例
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": 15
}
```

响应规范：
- 保持相同id用于请求追踪
- result字段返回执行结果（如15）
- SDK封装：通过装饰器自动处理参数传递，开发者只需关注业务逻辑实现


---

## A2A

- A2A 提供 了一种统一的封装方式。这样一来，不同来源的 Agent 能够实现互相调用，从而打破彼此之 间的隔阂，避免 Agent 成为孤立的**“信息孤岛”**，这对推动 Agent 之间的协同合作与生态 发展很有价值。
- Agent 分为主机 Agent 与远程 Agent。其中主机 Agent 仅负责管理与分发，而 远程 Agent 则是具体处理业务的 Agent，比如体育助手 Agent 等。
- 每一个独立的远程 Agent 都可以调用自身的工具，调用方式既可以是传统的函数调用，也可 以通过 MCP 实现；多个远程 Agent 可以使用 A2A 协议，通过一个主机 Agent 进行管理。

![](https://cdn.nlark.com/yuque/0/2026/png/12849168/1770368525751-a6b179b3-a4e7-4ca7-9ca9-668d0bcc5e4a.png)

A2A 所解决的核心问题是 Agent 如何进行标准化封装，从而实现多 Agent 协同 工作的能力，其抽象层级高于 MCP，二者构成了互补性的协议体系。