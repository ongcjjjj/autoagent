#!/usr/bin/env python3
"""
高级配置示例 - 自主进化Agent系统
展示系统的高级配置和优化技巧
"""

import asyncio
import sys
import os
from typing import Dict, Any

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    AutonomousEvolutionarySystem,
    BaseAgent, AgentRole, CommunicationProtocol
)


class AdvancedSystemConfig:
    """高级系统配置类"""
    
    def __init__(self):
        self.performance_configs = {
            'high_performance': {
                'learning_rate': 0.25,
                'temperature': 0.6,
                'exploration_rate': 0.3,
                'adaptation_speed': 0.4,
                'memory_limit': 200,
                'evaluation_threshold': 0.8
            },
            'balanced': {
                'learning_rate': 0.15,
                'temperature': 0.8,
                'exploration_rate': 0.4,
                'adaptation_speed': 0.2,
                'memory_limit': 100,
                'evaluation_threshold': 0.6
            },
            'conservative': {
                'learning_rate': 0.08,
                'temperature': 0.9,
                'exploration_rate': 0.5,
                'adaptation_speed': 0.1,
                'memory_limit': 50,
                'evaluation_threshold': 0.4
            }
        }
        
        self.role_specializations = {
            AgentRole.RESEARCHER: {
                'temperature': 0.9,  # 高创造性
                'exploration_rate': 0.6,  # 高探索性
                'focus_areas': ['information_gathering', 'pattern_analysis']
            },
            AgentRole.EXECUTOR: {
                'temperature': 0.5,  # 低创造性，高执行力
                'exploration_rate': 0.2,  # 低探索性
                'focus_areas': ['task_execution', 'result_delivery']
            },
            AgentRole.CRITIC: {
                'temperature': 0.3,  # 严格评判
                'exploration_rate': 0.1,  # 保守评估
                'focus_areas': ['quality_assessment', 'error_detection']
            },
            AgentRole.COORDINATOR: {
                'temperature': 0.7,  # 平衡决策
                'exploration_rate': 0.3,  # 适度探索
                'focus_areas': ['resource_allocation', 'conflict_resolution']
            },
            AgentRole.OPTIMIZER: {
                'temperature': 0.4,  # 理性优化
                'exploration_rate': 0.4,  # 适度探索新方法
                'focus_areas': ['performance_tuning', 'efficiency_improvement']
            }
        }
    
    def apply_performance_config(self, system: AutonomousEvolutionarySystem, config_name: str):
        """应用性能配置"""
        if config_name not in self.performance_configs:
            raise ValueError(f"未知配置: {config_name}")
        
        config = self.performance_configs[config_name]
        
        for agent in system.agents.values():
            agent.learning_rate = config['learning_rate']
            agent.temperature = config['temperature']
            agent.exploration_rate = config['exploration_rate']
            agent.adaptation_speed = config['adaptation_speed']
            # memory_limit 通过内存管理逻辑控制，不直接设置属性
        
        # evaluation_threshold 通过系统逻辑控制，不直接设置属性
        
        print(f"✅ 应用了 '{config_name}' 性能配置")
    
    def apply_role_specialization(self, system: AutonomousEvolutionarySystem):
        """应用角色专业化配置"""
        for agent in system.agents.values():
            if agent.role in self.role_specializations:
                spec = self.role_specializations[agent.role]
                agent.temperature = spec['temperature']
                agent.exploration_rate = spec['exploration_rate']
                
                # 专业化焦点通过Agent行为逻辑体现，不直接设置属性
        
        print("✅ 应用了角色专业化配置")


class DynamicParameterTuner:
    """动态参数调优器"""
    
    def __init__(self, system: AutonomousEvolutionarySystem):
        self.system = system
        self.performance_history = []
        self.tuning_strategies = {
            'aggressive': {'step_size': 0.1, 'threshold': 0.05},
            'moderate': {'step_size': 0.05, 'threshold': 0.03},
            'conservative': {'step_size': 0.02, 'threshold': 0.01}
        }
    
    async def auto_tune_parameters(self, strategy: str = 'moderate', max_iterations: int = 5):
        """自动调优参数"""
        print(f"\n🔧 开始自动参数调优 (策略: {strategy})")
        
        if strategy not in self.tuning_strategies:
            strategy = 'moderate'
        
        config = self.tuning_strategies[strategy]
        step_size = config['step_size']
        threshold = config['threshold']
        
        baseline_metrics = await self.system.evaluate_system_performance()
        baseline_score = baseline_metrics.composite_score
        best_score = baseline_score
        best_params = self.get_current_parameters()
        
        print(f"   基线性能: {baseline_score:.3f}")
        
        for iteration in range(max_iterations):
            print(f"\n   调优迭代 {iteration + 1}/{max_iterations}")
            
            # 尝试不同的参数调整
            improvements = []
            
            # 调整学习率
            await self.tune_learning_rate(step_size, improvements)
            
            # 调整温度参数
            await self.tune_temperature(step_size, improvements)
            
            # 调整探索率
            await self.tune_exploration_rate(step_size, improvements)
            
            # 选择最佳改进
            if improvements:
                best_improvement = max(improvements, key=lambda x: x['score'])
                if best_improvement['score'] > best_score + threshold:
                    self.apply_parameter_change(best_improvement)
                    best_score = best_improvement['score']
                    best_params = best_improvement['params']
                    print(f"     ✅ 参数优化: {best_improvement['param_name']} -> {best_improvement['new_value']:.3f}")
                    print(f"     📈 性能提升: {best_score:.3f} (+{best_score - baseline_score:.3f})")
                else:
                    print(f"     ⏸️  无显著改进，停止调优")
                    break
            else:
                print(f"     ⏸️  无可行改进，停止调优")
                break
        
        # 应用最佳参数
        self.apply_parameters(best_params)
        final_score = await self.evaluate_performance()
        
        print(f"\n🎯 调优完成:")
        print(f"   最终性能: {final_score:.3f}")
        print(f"   总体提升: {final_score - baseline_score:.3f}")
        
        return {
            'baseline_score': baseline_score,
            'final_score': final_score,
            'improvement': final_score - baseline_score,
            'best_params': best_params
        }
    
    async def tune_learning_rate(self, step_size: float, improvements: list):
        """调优学习率"""
        current_lr = self.get_average_learning_rate()
        
        for direction in [1, -1]:
            new_lr = max(0.01, min(0.5, current_lr + direction * step_size))
            await self.test_parameter_change('learning_rate', new_lr, improvements)
    
    async def tune_temperature(self, step_size: float, improvements: list):
        """调优温度参数"""
        current_temp = self.get_average_temperature()
        
        for direction in [1, -1]:
            new_temp = max(0.1, min(1.0, current_temp + direction * step_size))
            await self.test_parameter_change('temperature', new_temp, improvements)
    
    async def tune_exploration_rate(self, step_size: float, improvements: list):
        """调优探索率"""
        current_exp = self.get_average_exploration_rate()
        
        for direction in [1, -1]:
            new_exp = max(0.1, min(0.8, current_exp + direction * step_size))
            await self.test_parameter_change('exploration_rate', new_exp, improvements)
    
    async def test_parameter_change(self, param_name: str, new_value: float, improvements: list):
        """测试参数变化"""
        # 保存当前参数
        old_params = self.get_current_parameters()
        
        # 应用新参数
        self.set_parameter_for_all_agents(param_name, new_value)
        
        # 评估性能
        score = await self.evaluate_performance()
        
        # 记录结果
        improvements.append({
            'param_name': param_name,
            'new_value': new_value,
            'score': score,
            'params': self.get_current_parameters()
        })
        
        # 恢复参数
        self.apply_parameters(old_params)
    
    def get_current_parameters(self) -> dict:
        """获取当前参数"""
        if not self.system.agents:
            return {}
        
        agent = next(iter(self.system.agents.values()))
        return {
            'learning_rate': agent.learning_rate,
            'temperature': agent.temperature,
            'exploration_rate': agent.exploration_rate,
            'adaptation_speed': agent.adaptation_speed
        }
    
    def apply_parameters(self, params: dict):
        """应用参数配置"""
        for agent in self.system.agents.values():
            for param_name, value in params.items():
                if hasattr(agent, param_name):
                    setattr(agent, param_name, value)
    
    def apply_parameter_change(self, improvement: dict):
        """应用参数改变"""
        param_name = improvement['param_name']
        new_value = improvement['new_value']
        self.set_parameter_for_all_agents(param_name, new_value)
    
    def set_parameter_for_all_agents(self, param_name: str, value: float):
        """为所有Agent设置参数"""
        for agent in self.system.agents.values():
            if hasattr(agent, param_name):
                setattr(agent, param_name, value)
    
    def get_average_learning_rate(self) -> float:
        """获取平均学习率"""
        if not self.system.agents:
            return 0.15
        return sum(agent.learning_rate for agent in self.system.agents.values()) / len(self.system.agents)
    
    def get_average_temperature(self) -> float:
        """获取平均温度"""
        if not self.system.agents:
            return 0.8
        return sum(agent.temperature for agent in self.system.agents.values()) / len(self.system.agents)
    
    def get_average_exploration_rate(self) -> float:
        """获取平均探索率"""
        if not self.system.agents:
            return 0.4
        return sum(agent.exploration_rate for agent in self.system.agents.values()) / len(self.system.agents)
    
    async def evaluate_performance(self) -> float:
        """评估性能"""
        metrics = await self.system.evaluate_system_performance()
        return metrics.composite_score


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, system: AutonomousEvolutionarySystem):
        self.system = system
        self.monitoring_active = False
        self.performance_log = []
    
    async def start_monitoring(self, interval: float = 5.0):
        """开始性能监控"""
        self.monitoring_active = True
        print(f"📊 开始性能监控 (间隔: {interval}秒)")
        
        while self.monitoring_active:
            try:
                metrics = await self.system.evaluate_system_performance()
                
                log_entry = {
                    'timestamp': asyncio.get_event_loop().time(),
                    'composite_score': metrics.composite_score,
                    'agent_count': len(self.system.agents),
                    'total_actions': sum(len(agent.action_history) for agent in self.system.agents.values()),
                    'average_temperature': sum(agent.temperature for agent in self.system.agents.values()) / len(self.system.agents) if self.system.agents else 0,
                    'memory_usage': sum(len(agent.memory) for agent in self.system.agents.values())
                }
                
                self.performance_log.append(log_entry)
                
                # 输出监控信息
                print(f"📈 性能监控: 得分={metrics.composite_score:.3f}, "
                      f"Agent数={len(self.system.agents)}, "
                      f"总行动={log_entry['total_actions']}, "
                      f"平均温度={log_entry['average_temperature']:.2f}")
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ 监控错误: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        print("⏹️  性能监控已停止")
    
    def get_performance_summary(self) -> dict:
        """获取性能摘要"""
        if not self.performance_log:
            return {}
        
        scores = [entry['composite_score'] for entry in self.performance_log]
        
        return {
            'total_samples': len(self.performance_log),
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'score_trend': scores[-1] - scores[0] if len(scores) > 1 else 0,
            'final_score': scores[-1]
        }


async def advanced_config_demo():
    """高级配置演示"""
    print("🚀 高级配置演示 - 自主进化Agent系统")
    print("=" * 60)
    
    # 1. 创建系统和配置器
    system = AutonomousEvolutionarySystem()
    config = AdvancedSystemConfig()
    
    # 创建标准团队
    team = system.create_standard_team()
    print(f"✅ 创建了 {len(team)} 个Agent")
    
    # 2. 应用高性能配置
    print("\n🔧 步骤1: 应用高性能配置")
    config.apply_performance_config(system, 'high_performance')
    config.apply_role_specialization(system)
    
    # 显示配置后的参数
    print("   Agent参数状态:")
    for agent_id, agent in list(system.agents.items())[:3]:  # 只显示前3个
        print(f"     {agent_id}: LR={agent.learning_rate:.2f}, "
              f"T={agent.temperature:.2f}, E={agent.exploration_rate:.2f}")
    
    # 3. 运行基准测试
    print("\n📊 步骤2: 基准性能测试")
    baseline_result = await system.run_collaborative_task(
        goal="系统基准性能测试",
        max_cycles=2
    )
    
    baseline_score = baseline_result['final_metrics'].composite_score if baseline_result['final_metrics'] else 0.5
    print(f"   基准性能: {baseline_score:.3f}")
    
    # 4. 自动参数调优
    print("\n🎯 步骤3: 自动参数调优")
    tuner = DynamicParameterTuner(system)
    tuning_result = await tuner.auto_tune_parameters(strategy='moderate', max_iterations=3)
    
    print(f"   调优结果:")
    print(f"     基线: {tuning_result['baseline_score']:.3f}")
    print(f"     最终: {tuning_result['final_score']:.3f}")
    print(f"     提升: {tuning_result['improvement']:.3f}")
    
    # 5. 性能监控演示
    print("\n📈 步骤4: 性能监控演示")
    monitor = PerformanceMonitor(system)
    
    # 启动监控任务
    monitoring_task = asyncio.create_task(monitor.start_monitoring(interval=2.0))
    
    # 运行一些任务进行监控
    tasks = [
        "数据处理优化任务",
        "算法性能分析任务", 
        "系统架构评估任务"
    ]
    
    for task in tasks:
        print(f"   执行任务: {task}")
        await system.run_collaborative_task(goal=task, max_cycles=1)
        await asyncio.sleep(1)  # 让监控器记录数据
    
    # 停止监控
    monitor.stop_monitoring()
    monitoring_task.cancel()
    
    # 显示监控摘要
    summary = monitor.get_performance_summary()
    if summary:
        print(f"\n📊 监控摘要:")
        print(f"   样本数: {summary['total_samples']}")
        print(f"   平均得分: {summary['average_score']:.3f}")
        print(f"   最高得分: {summary['max_score']:.3f}")
        print(f"   得分趋势: {summary['score_trend']:+.3f}")
    
    # 6. 配置对比测试
    print("\n⚖️  步骤5: 配置对比测试")
    configs_to_test = ['conservative', 'balanced', 'high_performance']
    config_results = {}
    
    for config_name in configs_to_test:
        print(f"   测试配置: {config_name}")
        config.apply_performance_config(system, config_name)
        
        result = await system.run_collaborative_task(
            goal=f"配置测试任务 - {config_name}",
            max_cycles=1
        )
        
        score = result['final_metrics'].composite_score if result['final_metrics'] else 0.5
        config_results[config_name] = score
        print(f"     性能得分: {score:.3f}")
    
    # 显示最佳配置
    if config_results:
        best_config = max(config_results.keys(), key=lambda x: config_results[x])
        print(f"\n🏆 最佳配置: {best_config} (得分: {config_results[best_config]:.3f})")
    else:
        print(f"\n⚠️  无配置测试结果")
    
    print("\n🎉 高级配置演示完成！")


if __name__ == "__main__":
    print("⚙️  高级配置示例")
    print("展示系统的高级配置、自动调优和性能监控功能\n")
    
    asyncio.run(advanced_config_demo())
    
    print("\n" + "=" * 60)
    print("💡 高级特性总结:")
    print("1. 多种性能配置预设 (高性能/平衡/保守)")
    print("2. 角色专业化参数配置")
    print("3. 自动参数调优算法")
    print("4. 实时性能监控")
    print("5. 配置对比测试")
    print("6. 动态参数优化")
    print("=" * 60)