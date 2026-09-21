
AutoGen / CrewAI

## 微调

### 目的

微调（Fine-tuning）的核心目的，是在通用大模型（General-Purpose LLM）的基础上，通过小规模、特定领域的数据训练，使其行为更贴合具体业务场景的需求。

微调不是为了“让模型学会新知识”（那是 RAG 的任务），而是为了改变模型的“表达方式” 、 “决策偏好”和“输出风格” ，实现深度定制化。

![微调目的](images/Pasted%20image%2020260810225736.png)

### 微调方法

- LoRA：低秩适配，训练快，效果好
- P-Tuning v2：可学习的Prompt，适合任务特定优化
- Adapter：插入小模块，模块化更新

![微调方法](images/Pasted%20image%2020260810225838.png)

适用场景分析

- 数据量少（<1k条） → 优先使用 P-Tuning 或 Prompt Tuning
- 性能要求高、资源充足 → 选择 LoRA，平衡效果与部署成本
- 需模块化升级、多任务切换 → 使用 Adapter，支持热插拔
- 特定领域知识注入（如法律、医疗） → LoRA + RAG 联合使用
- 低延迟服务场景 → 避免 Adapter，优先 LoRA 或 P-Tuning
