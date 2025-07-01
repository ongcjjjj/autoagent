"""
智能任务执行引擎
实现任务分解、并行执行、依赖管理、智能调度、错误恢复
"""
import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskDependency:
    """任务依赖"""
    task_id: str
    depends_on: List[str]
    dependency_type: str = "completion"  # completion, success, data
    condition: Optional[Callable] = None

@dataclass
class Task:
    """任务定义"""
    task_id: str
    name: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 60.0  # 秒
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 状态跟踪
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.available_resources = {
            "cpu_cores": 4,
            "memory_mb": 8192,
            "network_connections": 100,
            "disk_space_mb": 10240
        }
        self.allocated_resources = defaultdict(float)
        self.resource_usage_history = deque(maxlen=1000)
        
    def can_allocate(self, resources_required: Dict[str, Any]) -> bool:
        """检查是否可以分配资源"""
        for resource, amount in resources_required.items():
            if resource in self.available_resources:
                available = self.available_resources[resource] - self.allocated_resources[resource]
                if available < amount:
                    return False
        return True
    
    def allocate(self, task_id: str, resources_required: Dict[str, Any]) -> bool:
        """分配资源"""
        if not self.can_allocate(resources_required):
            return False
        
        for resource, amount in resources_required.items():
            self.allocated_resources[resource] += amount
        
        self.resource_usage_history.append({
            "task_id": task_id,
            "resources": resources_required,
            "timestamp": time.time(),
            "action": "allocate"
        })
        
        return True
    
    def release(self, task_id: str, resources_required: Dict[str, Any]):
        """释放资源"""
        for resource, amount in resources_required.items():
            self.allocated_resources[resource] = max(0, self.allocated_resources[resource] - amount)
        
        self.resource_usage_history.append({
            "task_id": task_id,
            "resources": resources_required,
            "timestamp": time.time(),
            "action": "release"
        })
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        utilization = {}
        for resource, total in self.available_resources.items():
            used = self.allocated_resources[resource]
            utilization[resource] = used / total if total > 0 else 0.0
        return utilization

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.task_queue = deque()
        self.running_tasks = {}
        self.completed_tasks = {}
        self.dependency_graph = {}
        self.scheduling_policies = {
            "priority": self._priority_scheduling,
            "fifo": self._fifo_scheduling,
            "shortest_job_first": self._sjf_scheduling,
            "resource_aware": self._resource_aware_scheduling
        }
        self.current_policy = "resource_aware"
        
    def add_task(self, task: Task):
        """添加任务到调度队列"""
        self.task_queue.append(task)
        self._update_dependency_graph(task)
        logger.info(f"Task {task.task_id} added to scheduler")
    
    def _update_dependency_graph(self, task: Task):
        """更新依赖图"""
        self.dependency_graph[task.task_id] = task.dependencies
    
    def get_ready_tasks(self) -> List[Task]:
        """获取可执行的任务"""
        ready_tasks = []
        
        for task in list(self.task_queue):
            if task.status == TaskStatus.PENDING and self._dependencies_satisfied(task):
                task.status = TaskStatus.READY
                ready_tasks.append(task)
        
        # 应用调度策略
        if ready_tasks:
            policy_func = self.scheduling_policies.get(self.current_policy, self._fifo_scheduling)
            ready_tasks = policy_func(ready_tasks)
        
        return ready_tasks
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """检查任务依赖是否满足"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _priority_scheduling(self, tasks: List[Task]) -> List[Task]:
        """优先级调度"""
        return sorted(tasks, key=lambda t: t.priority.value, reverse=True)
    
    def _fifo_scheduling(self, tasks: List[Task]) -> List[Task]:
        """先进先出调度"""
        return sorted(tasks, key=lambda t: t.created_at)
    
    def _sjf_scheduling(self, tasks: List[Task]) -> List[Task]:
        """最短作业优先调度"""
        return sorted(tasks, key=lambda t: t.estimated_duration)
    
    def _resource_aware_scheduling(self, tasks: List[Task]) -> List[Task]:
        """资源感知调度"""
        # 按资源需求和优先级综合排序
        def scoring_func(task):
            priority_score = task.priority.value * 10
            resource_score = 0
            
            # 计算资源适配度
            if task.resources_required:
                total_required = sum(task.resources_required.values())
                if self.resource_manager.can_allocate(task.resources_required):
                    resource_score = 10 - min(total_required / 100, 10)
            
            return priority_score + resource_score
        
        return sorted(tasks, key=scoring_func, reverse=True)

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.execution_stats = {
            "total_executed": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "average_execution_time": 0.0
        }
        
    async def execute_task(self, task: Task, resource_manager: ResourceManager) -> TaskResult:
        """执行单个任务"""
        async with self.execution_semaphore:
            result = TaskResult(task_id=task.task_id, status=TaskStatus.RUNNING)
            
            try:
                # 分配资源
                if not resource_manager.allocate(task.task_id, task.resources_required):
                    result.status = TaskStatus.FAILED
                    result.error = "Resource allocation failed"
                    return result
                
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                result.start_time = task.started_at
                
                self.running_tasks[task.task_id] = task
                
                # 执行任务
                if asyncio.iscoroutinefunction(task.func):
                    # 异步函数
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            task.func(*task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    else:
                        task_result = await task.func(*task.args, **task.kwargs)
                else:
                    # 同步函数
                    if task.timeout:
                        task_result = await asyncio.wait_for(
                            asyncio.to_thread(task.func, *task.args, **task.kwargs),
                            timeout=task.timeout
                        )
                    else:
                        task_result = await asyncio.to_thread(task.func, *task.args, **task.kwargs)
                
                # 任务成功完成
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.end_time = task.completed_at
                result.execution_time = task.completed_at - task.started_at
                
                self.execution_stats["successful"] += 1
                
            except asyncio.TimeoutError:
                result.status = TaskStatus.FAILED
                result.error = f"Task timed out after {task.timeout} seconds"
                result.end_time = time.time()
                result.execution_time = result.end_time - result.start_time
                
                task.status = TaskStatus.FAILED
                self.execution_stats["failed"] += 1
                
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = str(e)
                result.end_time = time.time()
                result.execution_time = result.end_time - result.start_time
                
                task.status = TaskStatus.FAILED
                self.execution_stats["failed"] += 1
                
                logger.error(f"Task {task.task_id} failed: {e}")
                
            finally:
                # 释放资源
                resource_manager.release(task.task_id, task.resources_required)
                
                # 从运行列表中移除
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                
                # 更新统计
                self.execution_stats["total_executed"] += 1
                self._update_execution_stats(result.execution_time)
            
            return result
    
    def _update_execution_stats(self, execution_time: float):
        """更新执行统计"""
        total = self.execution_stats["total_executed"]
        current_avg = self.execution_stats["average_execution_time"]
        
        # 计算移动平均
        self.execution_stats["average_execution_time"] = (
            (current_avg * (total - 1) + execution_time) / total
        )

class ErrorRecoveryManager:
    """错误恢复管理器"""
    
    def __init__(self):
        self.retry_strategies = {
            "exponential_backoff": self._exponential_backoff_retry,
            "linear_backoff": self._linear_backoff_retry,
            "immediate": self._immediate_retry,
            "adaptive": self._adaptive_retry
        }
        self.error_patterns = defaultdict(int)
        self.recovery_history = deque(maxlen=100)
        
    async def handle_task_failure(
        self, 
        task: Task, 
        result: TaskResult, 
        scheduler: TaskScheduler
    ) -> bool:
        """处理任务失败"""
        # 记录错误模式
        error_type = type(result.error).__name__ if result.error else "UnknownError"
        self.error_patterns[error_type] += 1
        
        # 判断是否需要重试
        if result.retry_count >= task.max_retries:
            logger.warning(f"Task {task.task_id} exceeded max retries ({task.max_retries})")
            return False
        
        # 选择重试策略
        retry_strategy = self._select_retry_strategy(task, result)
        
        # 计算重试延迟
        delay = retry_strategy(result.retry_count + 1)
        
        # 记录恢复历史
        self.recovery_history.append({
            "task_id": task.task_id,
            "error_type": error_type,
            "retry_count": result.retry_count + 1,
            "delay": delay,
            "timestamp": time.time()
        })
        
        # 安排重试
        await asyncio.sleep(delay)
        
        # 重置任务状态
        task.status = TaskStatus.RETRYING
        result.retry_count += 1
        
        # 重新添加到调度队列
        scheduler.add_task(task)
        
        logger.info(f"Retrying task {task.task_id} (attempt {result.retry_count + 1})")
        return True
    
    def _select_retry_strategy(self, task: Task, result: TaskResult) -> Callable:
        """选择重试策略"""
        # 基于错误类型和任务特性选择策略
        if "timeout" in str(result.error).lower():
            return self.retry_strategies["exponential_backoff"]
        elif "resource" in str(result.error).lower():
            return self.retry_strategies["linear_backoff"]
        elif task.priority == TaskPriority.CRITICAL:
            return self.retry_strategies["immediate"]
        else:
            return self.retry_strategies["adaptive"]
    
    def _exponential_backoff_retry(self, attempt: int) -> float:
        """指数退避重试"""
        return min(2 ** attempt, 300)  # 最大5分钟
    
    def _linear_backoff_retry(self, attempt: int) -> float:
        """线性退避重试"""
        return min(attempt * 30, 300)  # 最大5分钟
    
    def _immediate_retry(self, attempt: int) -> float:
        """立即重试"""
        return 1.0
    
    def _adaptive_retry(self, attempt: int) -> float:
        """自适应重试"""
        # 基于历史成功率调整延迟
        base_delay = attempt * 10
        
        if len(self.recovery_history) > 10:
            recent_recoveries = list(self.recovery_history)[-10:]
            success_rate = len([r for r in recent_recoveries if r.get("success", False)]) / 10
            
            # 成功率高时减少延迟，成功率低时增加延迟
            multiplier = 2.0 - success_rate
            return min(base_delay * multiplier, 300)
        
        return base_delay

class TaskExecutionEngine:
    """任务执行引擎主类"""
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.resource_manager = ResourceManager()
        self.scheduler = TaskScheduler(self.resource_manager)
        self.executor = TaskExecutor(max_concurrent_tasks)
        self.error_recovery = ErrorRecoveryManager()
        
        self.tasks = {}
        self.results = {}
        self.execution_loop_running = False
        self.execution_loop_task = None
        
        self.performance_metrics = {
            "throughput": deque(maxlen=60),  # 每分钟吞吐量
            "latency": deque(maxlen=100),    # 任务延迟
            "resource_efficiency": deque(maxlen=60),
            "error_rate": deque(maxlen=60)
        }
        
    def submit_task(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: List[str] = None,
        **task_options
    ) -> str:
        """提交任务"""
        if kwargs is None:
            kwargs = {}
        if dependencies is None:
            dependencies = []
        
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies,
            **task_options
        )
        
        self.tasks[task_id] = task
        self.scheduler.add_task(task)
        
        logger.info(f"Submitted task {task_id}: {name}")
        return task_id
    
    def submit_workflow(self, workflow_definition: Dict[str, Any]) -> List[str]:
        """提交工作流"""
        task_ids = []
        
        # 解析工作流定义
        for step_name, step_config in workflow_definition.get("steps", {}).items():
            task_id = self.submit_task(
                name=f"workflow_step_{step_name}",
                func=step_config["func"],
                args=step_config.get("args", ()),
                kwargs=step_config.get("kwargs", {}),
                dependencies=step_config.get("dependencies", []),
                priority=TaskPriority(step_config.get("priority", 2))
            )
            task_ids.append(task_id)
        
        return task_ids
    
    async def start_execution_loop(self):
        """启动执行循环"""
        if self.execution_loop_running:
            return
        
        self.execution_loop_running = True
        self.execution_loop_task = asyncio.create_task(self._execution_loop())
        logger.info("Task execution loop started")
    
    async def stop_execution_loop(self):
        """停止执行循环"""
        self.execution_loop_running = False
        
        if self.execution_loop_task:
            self.execution_loop_task.cancel()
            try:
                await self.execution_loop_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task execution loop stopped")
    
    async def _execution_loop(self):
        """主执行循环"""
        while self.execution_loop_running:
            try:
                # 获取可执行的任务
                ready_tasks = self.scheduler.get_ready_tasks()
                
                if ready_tasks:
                    # 并行执行任务
                    execution_coroutines = [
                        self._execute_task_with_recovery(task)
                        for task in ready_tasks
                        if self.resource_manager.can_allocate(task.resources_required)
                    ]
                    
                    if execution_coroutines:
                        await asyncio.gather(*execution_coroutines, return_exceptions=True)
                
                # 更新性能指标
                self._update_performance_metrics()
                
                # 短暂休眠以避免CPU占用过高
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task_with_recovery(self, task: Task):
        """执行任务并处理错误恢复"""
        try:
            # 从队列中移除任务
            if task in self.scheduler.task_queue:
                self.scheduler.task_queue.remove(task)
            
            # 执行任务
            result = await self.executor.execute_task(task, self.resource_manager)
            
            # 存储结果
            self.results[task.task_id] = result
            
            if result.status == TaskStatus.COMPLETED:
                self.scheduler.completed_tasks[task.task_id] = result
                logger.info(f"Task {task.task_id} completed successfully")
            else:
                # 处理失败
                recovery_initiated = await self.error_recovery.handle_task_failure(
                    task, result, self.scheduler
                )
                
                if not recovery_initiated:
                    logger.error(f"Task {task.task_id} failed permanently: {result.error}")
            
        except Exception as e:
            logger.error(f"Unexpected error executing task {task.task_id}: {e}")
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        current_time = time.time()
        
        # 计算吞吐量 (每分钟完成的任务数)
        completed_last_minute = sum(
            1 for result in self.results.values()
            if result.end_time and (current_time - result.end_time) < 60
            and result.status == TaskStatus.COMPLETED
        )
        self.performance_metrics["throughput"].append(completed_last_minute)
        
        # 计算平均延迟
        recent_latencies = [
            result.execution_time for result in self.results.values()
            if result.execution_time > 0 and result.end_time
            and (current_time - result.end_time) < 300  # 最近5分钟
        ]
        if recent_latencies:
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            self.performance_metrics["latency"].append(avg_latency)
        
        # 计算资源效率
        utilization = self.resource_manager.get_resource_utilization()
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        self.performance_metrics["resource_efficiency"].append(avg_utilization)
        
        # 计算错误率
        total_recent = len([
            r for r in self.results.values()
            if r.end_time and (current_time - r.end_time) < 300
        ])
        failed_recent = len([
            r for r in self.results.values()
            if r.end_time and (current_time - r.end_time) < 300
            and r.status == TaskStatus.FAILED
        ])
        error_rate = failed_recent / total_recent if total_recent > 0 else 0
        self.performance_metrics["error_rate"].append(error_rate)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        result = self.results.get(task_id)
        
        return {
            "task_id": task_id,
            "name": task.name,
            "status": task.status.value,
            "priority": task.priority.value,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "execution_time": result.execution_time if result else None,
            "retry_count": result.retry_count if result else 0,
            "error": result.error if result else None,
            "dependencies": task.dependencies,
            "resources_required": task.resources_required
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        running_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        return {
            "execution_loop_running": self.execution_loop_running,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "running_tasks": running_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": total_tasks - completed_tasks - running_tasks - failed_tasks,
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "executor_stats": self.executor.execution_stats,
            "performance_metrics": {
                "avg_throughput": sum(self.performance_metrics["throughput"]) / len(self.performance_metrics["throughput"]) if self.performance_metrics["throughput"] else 0,
                "avg_latency": sum(self.performance_metrics["latency"]) / len(self.performance_metrics["latency"]) if self.performance_metrics["latency"] else 0,
                "avg_resource_efficiency": sum(self.performance_metrics["resource_efficiency"]) / len(self.performance_metrics["resource_efficiency"]) if self.performance_metrics["resource_efficiency"] else 0,
                "avg_error_rate": sum(self.performance_metrics["error_rate"]) / len(self.performance_metrics["error_rate"]) if self.performance_metrics["error_rate"] else 0
            }
        }
    
    async def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """等待任务完成"""
        start_time = time.time()
        
        while True:
            # 检查所有任务是否完成
            all_completed = True
            results = {}
            
            for task_id in task_ids:
                if task_id in self.results:
                    results[task_id] = self.results[task_id]
                    if self.results[task_id].status in [TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RETRYING]:
                        all_completed = False
                else:
                    all_completed = False
            
            if all_completed:
                return results
            
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Tasks did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.5)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.PENDING, TaskStatus.READY]:
            task.status = TaskStatus.CANCELLED
            # 从队列中移除
            if task in self.scheduler.task_queue:
                self.scheduler.task_queue.remove(task)
            return True
        
        return False  # 无法取消正在运行的任务