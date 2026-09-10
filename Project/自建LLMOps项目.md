
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

## 功能设计

### 聊天机器人

![[images/Pasted image 20260807155437.png]]

![[images/Pasted image 20260807155505.png]]

![[images/Pasted image 20260807155911.png]]


## 部署上线

### 安全策略

- 企业级安全措施
    - 路由平台设计：为大模型调用路由平台设计时，需要考虑不同模型的安全策略组合。
    - 多维度防护：综合运用开销控制（流量消耗监控）、时间控制（过期机制）和状态控制（开关禁用）等多重防护手段。
    - 实时监控：建立API调用监控系统，及时发现异常调用模式，如示例中通过打印调试信息监控密钥使用情况。


### 第三方社交媒体对接
![[images/Pasted image 20260907212733.png]]
![[images/Pasted image 20260907212756.png]]