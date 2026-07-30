 
## 1. 函数调用

**函数调用=大语言模型（支持函数调用）+预定Prompt+函数/工具参数列表+本地调用代码**
![[images/Pasted image 20260727220536.png]]

> 注：大语言模型本身的`函数调用`并不会调用我们预定义的`参数`，而是仅仅生成我们需要调用的函数的调用参数而已，具体调用函数的动作，需要我们在应用代码中实现

## 2. langchain中的工具与工具包

**内置工具**
langchain中封装了大量的**预设工具**，涵盖搜索、图像生成、百科、视频信息提取、代码执行、数据分析等。

所有工具的基类：BaseTool
如果需要将工具转为openAI工具形式参数，可以使用convert_to_openai_tool()辅助函数

**内置工具包**
工具包是一组设计用于一起执行特定任务的工具集，有get_tools()方法

## 3. 创建自定义工具

创建自定义工具三种方法：
1. `@tool`装饰器：适合已有简单函数的快速转换
2. `StructuredTool.form_function()`类方法，支持同步和异步，且配置项灵活
3. 继承`BaseTool`基类，并实现`_run（）`方法来实现自定义工具


## 4. chatModel使用函数调用

```python
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```
支持函数调用的大语言模型可以直接调用bind() 和 bind_tools()
![[images/Pasted image 20260728220402.png]]

附：**不支持函数调用的大语言模型解决思路？**
通过Prompt实现函数调用功能思路
![[images/Pasted image 20260728220434.png]]
## 5.结构化输出策略与选择

让 LLM 返回结构化输出非常重要，这是因为 LLM 应用程序的输出通常用于下游程序，这些应用程序需要特定的参数，目前常见的几种让 LLM 结构化输出的策略有：
1. Prompt：通过 prompt 让 LLM 输出特定结构的内容，兼容所有 LLM，但是输出不稳定。
2. 函数/工具调用：让 LLM 绑定函数，并设置选择模式为强制，让 LLM 强制调用函数，从而获取结构化输出数据。
3. JSON模式：对于支持 JSON 模式输出的 LLM，还可以通过设置输出结构为 JSON模式，从而获取结构化数据。

其中后两种输出模式会更稳定一些，在 LangChain 中为后两种方法封装了 `.with_structured_output()`方法，这也是获取结构化输出的最简单和最可靠的方法，在 `.with_structured_output()`的底层会使用 LLM 原生的函数调用或 JSON模式。

该方法接受一个 BaseModel子类 作为输入，该**子类需要指定输出属性的名称、描述和类型**，该方法返回的是一个 Runnable 可运行对象，**输出与给定模式对应的对象，其中模式可以指定为 JSON 模式（返回一个字典）或 Pydantic 类（返回一个 Pydantic 对象）**

附：**.with_structured_output的实现思路：**
构建一个虚假的函数，让 LLM 绑定该函数，并且设置 `tool_choice="any"`即强制调用所有函数，这样就可以在最大程度上确保 LLM 能稳定地输出结构化数据

## 6. 函数调用出错捕获提升程序健壮性

### try/except捕获工具错误

在工具调用步骤上使用 try/except 进行错误的捕获，并在出错时返回有用的提示信息，修正代码部分如下
```python
def try_except_tool(tool_args: dict, config: RunnableConfig) -> Any:
    try:
        return complex_tool.invoke(tool_args, config)
    except Exception as e:
        return f"调用工具时使用以下参数:\n\n{tool_args}\n\n引发了以下错误:\n\n{type(e)}: {e}"

chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool
```
在这种模式下，如果将工具返回的错误消息重新返回给 LLM 时，LLM 会重新判断并继续执行函数调用，从而将参数补全，让程序正常执行，也是最推荐的一种错误处理方案，即在工具内部进行错误的捕获，并将错误信息独立返回。
### 回退与重试处理

用with_fallback、with_retry

在函数调用中也可以使用这两种策略来处理，例如：
1. 在函数调用参数生成错误时，可以考虑回退到一个更好的模型，
2. 携带错误信息的重试策略：利用LLM 强大的自然语言处理功能进行自我纠正，将错误信息一起给LLM，让其重新操作

## 7. 多模态LLM执行函数调用

多模态LLM想**调用函数**工具，只需要按照正常的方式将工具绑定到模型上，在传递**消息**给LLM时，按照**多模态LLM**的特定规则即可