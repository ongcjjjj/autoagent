# 🔧 API参考文档

## 目录

- [核心系统](#核心系统)
- [Agent类](#agent类)
- [通信协议](#通信协议)
- [评估系统](#评估系统)
- [工具函数](#工具函数)
- [配置参数](#配置参数)

## 核心系统

### AutonomousEvolutionarySystem

主要的系统类，管理所有Agent和系统功能。

#### 构造函数

```python
AutonomousEvolutionarySystem()
```

创建一个新的自主进化系统实例。

#### 主要方法

##### `create_standard_team()`

```python
def create_standard_team() -> Dict[str, BaseAgent]
```

创建标准的5人Agent团队。

**返回值:**
- `Dict[str, BaseAgent]`: 包含5个专业化Agent的字典

**示例:**
```python
system = AutonomousEvolutionarySystem()
team = system.create_standard_team()
print(f"创建了 {len(team)} 个Agent")
```

##### `add_agent(agent)`

```python
def add_agent(self, agent: BaseAgent) -> None
```

向系统添加一个Agent。

**参数:**
- `agent`: BaseAgent实例

**示例:**
```python
researcher = ResearcherAgent("custom_researcher", communication)
system.add_agent(researcher)
```

##### `run_collaborative_task(goal, max_cycles)`

```python
async def run_collaborative_task(
    self, 
    goal: str, 
    max_cycles: int = 10
) -> Dict[str, Any]
```

运行协作任务。

**参数:**
- `goal`: 任务目标描述
- `max_cycles`: 最大执行周期数

**返回值:**
- `Dict[str, Any]`: 包含执行结果的字典

**示例:**
```python
result = await system.run_collaborative_task(
    goal="分析用户行为数据",
    max_cycles=5
)
```

##### `evaluate_system_performance()`

```python
async def evaluate_system_performance() -> PerformanceMetrics
```

评估系统整体性能。

**返回值:**
- `PerformanceMetrics`: 性能指标对象

##### `save_system_state(filepath)`

```python
def save_system_state(self, filepath: str) -> None
```

保存系统状态到文件。

**参数:**
- `filepath`: 保存路径

##### `load_system_state(filepath)`

```python
def load_system_state(self, filepath: str) -> None
```

从文件加载系统状态。

**参数:**
- `filepath`: 文件路径

## Agent类

### BaseAgent (抽象基类)

所有Agent的基类，定义了Agent的基本接口。

#### 构造函数

```python
BaseAgent(agent_id: str, role: AgentRole, communication: CommunicationProtocol)
```

**参数:**
- `agent_id`: Agent唯一标识符
- `role`: Agent角色枚举
- `communication`: 通信协议实例

#### 抽象方法

##### `think(context)`

```python
async def think(self, context: Dict[str, Any]) -> Dict[str, Any]
```

Agent思考过程，分析当前情况并制定计划。

**参数:**
- `context`: 上下文信息

**返回值:**
- `Dict[str, Any]`: 行动计划

##### `act(plan)`

```python
async def act(self, plan: Dict[str, Any]) -> AgentAction
```

执行行动计划。

**参数:**
- `plan`: 行动计划

**返回值:**
- `AgentAction`: 行动结果

##### `observe(action_result)`

```python
async def observe(self, action_result: AgentAction) -> Dict[str, Any]
```

观察行动结果并学习。

**参数:**
- `action_result`: 行动结果

**返回值:**
- `Dict[str, Any]`: 观察结果

#### 实例属性

- `agent_id: str` - Agent标识符
- `role: AgentRole` - Agent角色
- `communication: CommunicationProtocol` - 通信协议
- `memory: List[Dict]` - 记忆存储
- `action_history: List[AgentAction]` - 行动历史
- `success_patterns: List[Dict]` - 成功模式
- `performance_history: List[Dict]` - 性能历史
- `learning_rate: float` - 学习率 (0.05-0.3)
- `temperature: float` - 创造性温度 (0.1-1.0)
- `exploration_rate: float` - 探索率 (0.1-0.8)
- `adaptation_speed: float` - 适应速度 (0.05-0.5)
- `optimization_counter: int` - 优化计数器

#### 实例方法

##### `add_memory(content, importance)`

```python
def add_memory(self, content: Any, importance: float = 0.5) -> None
```

添加记忆。

**参数:**
- `content`: 记忆内容
- `importance`: 重要性 (0-1)

##### `learn_from_success(action)`

```python
def learn_from_success(self, action: AgentAction) -> None
```

从成功行动中学习。

**参数:**
- `action`: 成功的行动

##### `record_performance(metrics)`

```python
def record_performance(self, metrics: Dict[str, Any]) -> None
```

记录性能指标。

**参数:**
- `metrics`: 性能指标字典

### 专业化Agent类

#### ResearcherAgent

研究者Agent，专注于信息收集和分析。

```python
ResearcherAgent(agent_id: str, communication: CommunicationProtocol)
```

**特点:**
- 高创造性 (temperature = 0.9)
- 强探索能力 (exploration_rate = 0.6)
- 专长信息收集和模式分析

#### ExecutorAgent

执行者Agent，专注于任务执行。

```python
ExecutorAgent(agent_id: str, communication: CommunicationProtocol)
```

**特点:**
- 高执行力 (temperature = 0.5)
- 低探索性 (exploration_rate = 0.2)
- 专长任务执行和结果交付

#### CriticAgent

评判者Agent，专注于质量评估。

```python
CriticAgent(agent_id: str, communication: CommunicationProtocol)
```

**特点:**
- 严格评判 (temperature = 0.3)
- 保守评估 (exploration_rate = 0.1)
- 专长质量评估和错误检测

#### CoordinatorAgent

协调者Agent，专注于任务协调。

```python
CoordinatorAgent(agent_id: str, communication: CommunicationProtocol)
```

**特点:**
- 平衡决策 (temperature = 0.7)
- 适度探索 (exploration_rate = 0.3)
- 专长资源分配和冲突解决

#### ArchitectAgent

架构师Agent，专注于系统优化。

```python
ArchitectAgent(agent_id: str, communication: CommunicationProtocol)
```

**特点:**
- 理性优化 (temperature = 0.4)
- 适度探索新方法 (exploration_rate = 0.4)
- 专长性能调优和效率改进

## 通信协议

### CommunicationProtocol

Agent间通信的协议类。

#### 构造函数

```python
CommunicationProtocol()
```

#### 主要方法

##### `publish(topic, message, sender)`

```python
def publish(self, topic: str, message: Any, sender: str) -> None
```

发布消息到指定主题。

**参数:**
- `topic`: 主题名称
- `message`: 消息内容
- `sender`: 发送者ID

##### `subscribe(topic, subscriber)`

```python
def subscribe(self, topic: str, subscriber: str) -> None
```

订阅主题。

**参数:**
- `topic`: 主题名称
- `subscriber`: 订阅者ID

##### `get_messages(topic)`

```python
def get_messages(self, topic: str) -> List[Dict[str, Any]]
```

获取主题的所有消息。

**参数:**
- `topic`: 主题名称

**返回值:**
- `List[Dict[str, Any]]`: 消息列表

## 评估系统

### PerformanceMetrics

性能指标类，包含9个维度的评估结果。

#### 属性

- `trainability: float` - 可训练性 (0-1)
- `generalization: float` - 泛化能力 (0-1)
- `expressiveness: float` - 表达能力 (0-1)
- `creativity_score: float` - 创造性得分 (0-1)
- `adaptation_rate: float` - 适应速度 (0-1)
- `collaboration_efficiency: float` - 协作效率 (0-1)
- `error_recovery_rate: float` - 错误恢复率 (0-1)
- `knowledge_retention: float` - 知识保持率 (0-1)
- `innovation_index: float` - 创新指数 (0-1)
- `composite_score: float` - 综合得分 (0-1)

#### 方法

##### `to_dict()`

```python
def to_dict(self) -> Dict[str, float]
```

转换为字典格式。

##### `from_dict(data)`

```python
@classmethod
def from_dict(cls, data: Dict[str, float]) -> 'PerformanceMetrics'
```

从字典创建实例。

### TrainingFreeEvaluator

训练无关评估器，无需训练即可评估模型性能。

#### 主要方法

##### `evaluate_trainability(model_params)`

```python
def evaluate_trainability(self, model_params: Dict) -> float
```

评估模型可训练性。

##### `evaluate_generalization(model_complexity)`

```python
def evaluate_generalization(self, model_complexity: float) -> float
```

评估泛化能力。

##### `evaluate_expressiveness(architecture_info)`

```python
def evaluate_expressiveness(self, architecture_info: Dict) -> float
```

评估表达能力。

## 工具函数

### 数据结构

#### AgentAction

Agent行动的数据结构。

```python
@dataclass
class AgentAction:
    agent_id: str
    action_type: ActionType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### ActionType

行动类型枚举。

```python
class ActionType(Enum):
    THINK = "think"
    EXECUTE = "execute"
    COMMUNICATE = "communicate"
    LEARN = "learn"
    OPTIMIZE = "optimize"
```

#### AgentRole

Agent角色枚举。

```python
class AgentRole(Enum):
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    CRITIC = "critic"
    COORDINATOR = "coordinator"
    OPTIMIZER = "optimizer"
```

## 配置参数

### 系统级配置

- `MAX_MEMORY_SIZE: int = 100` - 最大记忆容量
- `PERFORMANCE_HISTORY_SIZE: int = 50` - 性能历史记录大小
- `SUCCESS_PATTERN_LIMIT: int = 20` - 成功模式存储限制
- `OPTIMIZATION_THRESHOLD: float = 0.1` - 优化触发阈值
- `COLLABORATION_TIMEOUT: float = 30.0` - 协作超时时间

### Agent参数范围

- `learning_rate`: 0.05 - 0.3
- `temperature`: 0.1 - 1.0
- `exploration_rate`: 0.1 - 0.8
- `adaptation_speed`: 0.05 - 0.5

### 评估权重

```python
EVALUATION_WEIGHTS = {
    'trainability': 0.15,
    'generalization': 0.15,
    'expressiveness': 0.10,
    'creativity_score': 0.15,
    'adaptation_rate': 0.10,
    'collaboration_efficiency': 0.15,
    'error_recovery_rate': 0.10,
    'knowledge_retention': 0.05,
    'innovation_index': 0.05
}
```

## 使用示例

### 基础使用

```python
import asyncio
from autonomous_evolutionary_agent_system import AutonomousEvolutionarySystem

async def main():
    # 创建系统
    system = AutonomousEvolutionarySystem()
    
    # 创建团队
    team = system.create_standard_team()
    
    # 运行任务
    result = await system.run_collaborative_task(
        goal="优化数据处理流程",
        max_cycles=5
    )
    
    print(f"任务完成，得分: {result['final_metrics'].composite_score:.3f}")

asyncio.run(main())
```

### 自定义Agent

```python
from autonomous_evolutionary_agent_system import BaseAgent, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, agent_id, communication):
        super().__init__(agent_id, AgentRole.RESEARCHER, communication)
    
    async def think(self, context):
        return {"action_type": "custom_analysis"}
    
    async def act(self, plan):
        # 实现自定义行为
        pass
    
    async def observe(self, action_result):
        return {"success_score": 0.8}
```

### 高级配置

```python
# 配置高性能模式
for agent in system.agents.values():
    agent.learning_rate = 0.25
    agent.temperature = 0.6
    agent.exploration_rate = 0.3

# 运行性能评估
metrics = await system.evaluate_system_performance()
print(f"系统性能: {metrics.composite_score:.3f}")
```

## 错误处理

### 常见异常

- `AgentNotFoundError`: Agent未找到
- `CommunicationError`: 通信错误
- `EvaluationError`: 评估错误
- `SystemStateError`: 系统状态错误

### 异常处理示例

```python
try:
    result = await system.run_collaborative_task(goal="test")
except Exception as e:
    print(f"任务执行失败: {e}")
    # 恢复到安全状态
    system.reset_to_safe_state()
```

---

**注意**: 所有异步方法都需要在异步环境中调用，建议使用 `asyncio.run()` 或在已有的异步函数中调用。