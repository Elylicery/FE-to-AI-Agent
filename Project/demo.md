
## 1. 聊天机器人
### 代码实现

```python

from openai import OpenAI

# 替换为你自己的 API Key
client = OpenAI(api_key="sk-xxx")

messages = [
    {"role": "system", "content": "你是一个有用的助手。"}
]

def chat(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    )

    reply = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            reply += delta
    print()

    messages.append({"role": "assistant", "content": reply})
    return reply
```

### 其他技术点

#### 将.env加载到环境变量实现配置分离

使用python-dotenv库。通过os.getEnv获取，**代码与配置分离**

#### 接口请求校验

为什么校验？
* 安全性：确保输入符合语气，避免安全漏洞，如sql注入等
* 数据完整性：例如数据格式、类型、长度等，避免因为不完整或不正确数据导致系统出错
* 提高用户体验：对不合法请求给出友好提示，提升用户体验

如何做？

**flash-wtf**（基于wtforms封装的Flask插件）支持 **CSRF保护**、快速提取数据、自定义验证规则登
 
#### 统一响应接口设计与实现
```
{ "code": "success", // 响应状态
"message": "获取AI应用数据成功", // 业务消息提示
"data": {} // 业务获取的数据
}
```
包含异常错误状态设计
#### Pytest与API测试用例编写

pytest：一个功能强大且易于使用的python测试框架，用fixture为测试提供预设数据和设置测试环境的功能。用`@pytest_mark_parametrize`装饰测试函数
pytest.ini配置运行时
#### Flask-SQLAlchemy扩展

Flask-SQLAIChemy是一个基于SQLAlchemy的FLask扩展


#### 应用ORM模型的CRUD

ORM是对象映射关系（object-Relational Mapping）即将数据库中的表与面向对象编程中的类关联起来，通过对类的操作来进行增删改查
