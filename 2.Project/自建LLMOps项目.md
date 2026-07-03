项目核心：可视化编排+智能化定制+多LLM接口支持+多平台对接

## LLMOps基础

### 为什么需要？

核心需求: 是否存在可视化平台能让非编程人员通过拖拽模块+填写提示词快速创建一个AI Agent应用并调试部署？

- **LLMOps定义**: **基于LLM的应用程序生命周期管理平台或者工具**，涵盖开发、部署、配置、运维全流程
- LLMOps旨在 **优化和简化LLM应用程序的各个环节**，以确保LLM应用高效、可靠和安全地运行
- LLMOps对使用者友好，**极大降低了企业创建AI Agent应用的成本**，把复杂的部分留给了LLMOps开发者


### 平台价值
 
 **LLMOps平台在AI应用开发环节中起什么作用？**

AI应用开发各环节差异

| 步骤        | 未使用LLMOps平台             | 使用LLMOps平台                          | 时间差异 |
| :-------- | :---------------------- | :---------------------------------- | :--- |
| 开发应用前&后端  | 集成和封装LLM能力，花费较多时间开发前端应用 | 直接使用LLMOps的后端服务，可基于开放API、WebApp直接开发 | -80% |
| 提示工程      | 仅能通过API或Playground进行    | Prompt可视化编排，所见即所得完成调试               | -25% |
| 数据准备与嵌入   | 编写代码实现长文本数据处理、嵌入        | 在LLMOps平台上传文本、文件                    | -80% |
| 应用日志与分析   | 编写代码查看日志，访问数据库查看        | 平台提供实时日志与分析                         | -70% |
| AI插件开发与集成 | 编写代码创建、集成AI插件           | 平台提供可视化工具创建、快速集成自定义插件能力             | -50% |
| AI工作流开发   | 编写代码完成每一个工作流            | 可视化编排工作流，所见即所得调试                    | -80% |

**两种模式开发AI应用流程对比**

- **不使用LLMOps**：整理需求、Prompt编写、部署LLM、对接LLM接口、处理应用数据、记录日志、应用工具开发、AI工作流开发、前后端交互开发、前后端部署。
- **使用LLMOps**：需求整理、Prompt编写、上传数据、勾选关联插件、可视化编排工作流、选择LLM模型、发布。

## 项目架构设计

![[images/Pasted image 20260627220740.png]]

### 项目目录结构
```
|---app // 应用入口集合
| ├---__init__.py
| └---http
|---config // 应用配置文件
| ├---__init__.py
| ├---config.py
| └---default_config.py
|---internal // 应用所有内部文件夹
| ├---core // LLM核心文件，集成LangChain、LLM、Embedding等非逻辑的代码
| | |---agent
| | |---chain
| | |---prompt
| | |---model_runtime
| | |---moderation
| | |---tool
| | |---vector_store
| | └---...
| ├---exception // 通用公共异常目录
| | ├---__init__.py
| | ├---exception.py
| | └---...
| ├---extension // Flask扩展文件目录
| | ├---__init__.py
| | ├---database_extension.py
| | └---...
| ├---handler // 路由处理器、控制器目录
| | ├---__init__.py
| | ├---account_handler.py
| | └---...
| ├---middleware // 应用中间件目录，包含校验是否登录
| | ├---__init__.py
| | └---middleware.py
| | └---...
| ├---migration // 数据库迁移文件目录，自动生成
| | ├---versions
| | └---...
| ├---model // 数据库模型文件目录
| | ├---__init__.py
| | ├---account.py
| | └---...
| ├---router // 应用路由文件夹
| | ├---__init__.py
| | ├---router.py
| | └---...
| ├---schedule // 调度任务、定时任务文件夹
| | ├---__init__.py
| | └---...
| ├---schema // 请求和响应的结构体
| | ├---__init__.py
| | └---...
| ├---server // 构建的应用，与app文件夹对应
| | ├---__init__.py
| | └---...
| ├---service // 服务层文件夹
| | ├---__init__.py
| | ├---oauth_service.py
| | └---...
| ├---task // 任务文件夹，支持即时任务+延迟任务
| | ├---__init__.py
| | └---...
|---pkg // 扩展包文件夹
| ├---__init__.py
| |---oauth
| | ├---__init__.py
| | ├---github_oauth.py
| | └---...
| └---... ├---storage // 本地存储文件夹
├---test // 测试目录
├---venv // 虚拟环境
├---.env // 应用配置文件
├---.gitignore // 配置git忽略文件
├---requirements.txt // 第三方包依赖管理
└---README.md // 项目说明文件
```

### 代码运行流程图

项目通过路由接收用户的发起的请求，并调用控制器特定的方法来处理，控制器接收到数据后，对数据进行校验，校验未通过则抛出错误；校验通过后将数据传递给Service/Core层进行相应的逻辑处理、数据存储和检索等操作，完成逻辑计算后得到响应数据，返回给控制器，控制器在将数据响应给用户，至此，一个最简单的流程结束。
![[images/Pasted image 20260629223123.png]]

### 技术选型

#### 数据库Postgres

PostgreSQL（又称 Postgres）是一种强大、开源的关系型数据库管理系统（RDBMS）

写一个批处理做windows下的启动与停止命令，会自动判断 PostgreSQL 的状态：正在运行就关闭，没运行就启动。

pg-toggle.bat
```bash
@echo off
chcp 65001 >nul
title PostgreSQL 16 管理

set PG_BIN=D:\softwareInstall\PostgreSQL\16\bin
set PG_DATA=D:\softwareInstall\PostgreSQL\16\data

echo 正在检测 PostgreSQL 运行状态...
"%PG_BIN%\pg_ctl.exe" status -D "%PG_DATA%" >nul 2>&1

if %errorlevel% equ 0 (
    echo PostgreSQL 正在运行，正在关闭...
    "%PG_BIN%\pg_ctl.exe" stop -D "%PG_DATA%"
) else (
    echo PostgreSQL 未运行，正在启动...
    "%PG_BIN%\pg_ctl.exe" start -D "%PG_DATA%"
)

echo.
pause

```
