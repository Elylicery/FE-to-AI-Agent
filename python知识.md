#### 依赖注入

- **定义**：依赖注入是软件工程中的一种**设计模式**，它允许在创建对象时由外部提供依赖关系，而不是自己创建这些依赖关系。
- **通俗解释**：简单来说，**即我需要什么，传递什么**，而不是自己内部构建。例如，controller需要使用service的内容，将service作为参数传递给controller即可。
-

**Python中的依赖注入框架injector**
* Injector是一个Python的轻量级依赖注入框架，通过装饰器 **@inject和Injector实例** 轻松实现依赖注入功能，不需要显式创建依赖对象。
```python
from injector import Injector, inject

class A:
    pass

@inject
class B:
    def __init__(self, a: A):
        self.a = a

injector = Injector()
b_instance = injector.get(B)
```