# 自我进化Agent

一个具备自我学习和进化能力的AI Agent，支持自定义OpenAI API配置，能够通过交互不断改进自身表现。

## 主要特性

### 🧠 智能记忆系统
- **短期记忆**: 存储最近的对话历史
- **长期记忆**: 保存重要的知识和经验
- **记忆检索**: 基于重要性和相关性的智能检索
- **记忆管理**: 自动清理过期记忆，保持最优状态

### 🧬 自我进化机制
- **性能评估**: 实时监控交互质量和效果
- **自动优化**: 根据表现数据自动调整策略
- **进化追踪**: 记录每次进化的改进点和策略
- **适应性学习**: 根据用户反馈持续改进

### ⚙️ 灵活配置
- **多API支持**: 兼容OpenAI API和兼容接口
- **自定义模型**: 支持GPT-3.5、GPT-4等多种模型
- **参数调优**: 可调整温度、token限制等参数
- **个性化设置**: 支持不同的交互风格和行为模式

### 🔧 开发友好
- **模块化设计**: 清晰的代码结构，易于扩展
- **丰富接口**: 提供命令行和API接口
- **数据导出**: 支持记忆和配置数据的导出导入
- **日志记录**: 详细的操作日志和性能监控

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

复制环境变量示例文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，设置你的API配置：
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. 启动Agent

#### 命令行模式
```bash
python main.py
```

#### 指定配置启动
```bash
python main.py --api-key your_key --model gpt-4 --name MyAgent
```

### 4. 开始使用

启动后进入交互模式，你可以：

- 直接与Agent对话
- 使用 `/help` 查看所有可用命令
- 使用 `/status` 查看Agent状态
- 使用 `/config` 管理配置
- 使用 `/memory search <query>` 搜索记忆
- 使用 `/evolution` 查看进化历史

## 配置选项

### OpenAI API 配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `api_key` | API密钥 | 必填 |
| `base_url` | API基础URL | `https://api.openai.com/v1` |
| `model` | 使用的模型 | `gpt-3.5-turbo` |
| `max_tokens` | 最大token数 | `2000` |
| `temperature` | 生成温度 | `0.7` |
| `timeout` | 请求超时时间 | `30` |

### Agent 配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `name` | Agent名称 | `SelfEvolvingAgent` |
| `memory_limit` | 记忆限制 | `1000` |
| `learning_rate` | 学习率 | `0.1` |
| `evolution_threshold` | 进化阈值 | `10` |

## 核心模块

### config.py
配置管理模块，支持环境变量和配置文件。

### memory.py
记忆管理模块，实现SQLite数据库存储的记忆系统。

### evolution.py
进化机制模块，实现性能评估和自动优化。

### openai_client.py
OpenAI API客户端，封装所有API调用。

### agent.py
主Agent类，集成所有功能模块。

### main.py
程序入口，提供命令行界面。

## API 使用示例

```python
from agent import SelfEvolvingAgent
import asyncio

async def example():
    # 创建Agent实例
    agent = SelfEvolvingAgent(name="MyAgent")
    
    # 测试连接
    if await agent.test_connection():
        print("连接成功！")
    
    # 处理消息
    response = await agent.process_message("你好，请介绍一下自己")
    print(response["content"])
    
    # 搜索记忆
    memories = agent.search_memory("介绍")
    print(f"找到 {len(memories)} 条相关记忆")
    
    # 查看进化历史
    evolution = agent.get_evolution_history()
    print(f"已进化 {len(evolution)} 次")

# 运行示例
asyncio.run(example())
```

## 进化机制详解

Agent的自我进化基于以下指标：

1. **成功率**: 任务完成的成功比例
2. **响应质量**: 回答的准确性和有用性
3. **学习效率**: 从交互中学习的速度
4. **适应速度**: 对新情况的适应能力
5. **用户满意度**: 基于用户反馈的满意程度

当这些指标低于阈值时，Agent会自动：
- 分析表现不佳的原因
- 制定针对性的改进策略
- 更新内部参数和行为模式
- 记录进化过程用于未来参考

## 数据存储

- **配置文件**: `agent_config.json`
- **记忆数据库**: `agent_memory.db`
- **进化数据**: `evolution_data.json`
- **个性设置**: `personality.json`

## 扩展开发

系统采用模块化设计，你可以轻松扩展：

1. **新的记忆类型**: 在 `memory.py` 中添加新的记忆类别
2. **自定义进化策略**: 在 `evolution.py` 中实现新的优化算法
3. **额外的API接口**: 在 `openai_client.py` 中添加新的API调用
4. **个性化行为**: 修改 `agent.py` 中的行为逻辑

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交Issue和Pull Request来帮助改进这个项目！

## 注意事项

1. 请确保API密钥的安全，不要提交到版本控制系统
2. 大量使用可能产生API费用，请注意监控使用量
3. 数据库文件包含所有对话记录，请妥善保管
4. 定期导出重要数据以防数据丢失

## 联系方式

如有问题或建议，请通过Issue联系我们。