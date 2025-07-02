#!/usr/bin/env python3
"""
自定义Agent示例 - 自主进化Agent系统
展示如何创建和使用自定义Agent
"""

import asyncio
import sys
import os
from typing import Dict, Any

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    BaseAgent, AgentAction, ActionType, AgentRole,
    AutonomousEvolutionarySystem, CommunicationProtocol
)
import time


class DataAnalystAgent(BaseAgent):
    """数据分析师Agent - 专门处理数据分析任务"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.RESEARCHER, communication)
        self.analysis_methods = [
            'statistical_analysis',
            'pattern_recognition', 
            'trend_analysis',
            'correlation_analysis'
        ]
        self.data_types = ['numerical', 'categorical', 'time_series', 'text']
        
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """数据分析师的思考过程"""
        goal = context.get('goal', 'general_analysis')
        
        # 分析任务类型
        if 'data' in goal.lower() or 'analysis' in goal.lower():
            analysis_type = 'data_analysis'
            complexity = 0.7
        elif 'pattern' in goal.lower() or 'trend' in goal.lower():
            analysis_type = 'pattern_analysis'
            complexity = 0.8
        else:
            analysis_type = 'general_analysis'
            complexity = 0.5
        
        plan = {
            'action_type': 'data_analysis',
            'analysis_type': analysis_type,
            'methods': self.select_analysis_methods(analysis_type),
            'estimated_complexity': complexity,
            'data_requirements': self.identify_data_requirements(goal)
        }
        
        return plan
    
    def select_analysis_methods(self, analysis_type: str) -> list:
        """选择分析方法"""
        if analysis_type == 'data_analysis':
            return ['statistical_analysis', 'correlation_analysis']
        elif analysis_type == 'pattern_analysis':
            return ['pattern_recognition', 'trend_analysis']
        else:
            return ['statistical_analysis']
    
    def identify_data_requirements(self, goal: str) -> dict:
        """识别数据需求"""
        requirements = {
            'data_types': [],
            'sample_size': 'medium',
            'quality_requirements': 'high'
        }
        
        if 'numerical' in goal.lower():
            requirements['data_types'].append('numerical')
        if 'text' in goal.lower():
            requirements['data_types'].append('text')
        if not requirements['data_types']:
            requirements['data_types'] = ['numerical']  # 默认
            
        return requirements
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行数据分析"""
        start_time = time.time()
        
        analysis_type = plan.get('analysis_type', 'general_analysis')
        methods = plan.get('methods', [])
        
        # 模拟数据分析过程
        analysis_results = {
            'analysis_type': analysis_type,
            'methods_used': methods,
            'findings': self.generate_findings(analysis_type),
            'confidence': self.calculate_confidence(plan),
            'recommendations': self.generate_recommendations(analysis_type)
        }
        
        # 广播分析结果
        self.communication.publish(
            'data_analysis_results',
            analysis_results,
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content=analysis_results,
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': analysis_results['confidence'],
                'analysis_complexity': plan.get('estimated_complexity', 0.5)
            }
        )
        
        self.action_history.append(action)
        return action
    
    def generate_findings(self, analysis_type: str) -> list:
        """生成分析发现"""
        findings = []
        
        if analysis_type == 'data_analysis':
            findings = [
                "发现数据中存在明显的线性趋势",
                "识别出3个主要的数据聚类",
                "检测到2个异常值需要进一步调查"
            ]
        elif analysis_type == 'pattern_analysis':
            findings = [
                "识别出周期性模式，周期约为7天",
                "发现季节性变化趋势",
                "检测到新兴模式在最近数据中出现"
            ]
        else:
            findings = [
                "数据质量良好，完整性达到95%",
                "基本统计特征符合预期范围"
            ]
        
        return findings
    
    def calculate_confidence(self, plan: Dict[str, Any]) -> float:
        """计算分析置信度"""
        base_confidence = 0.7
        complexity = plan.get('estimated_complexity', 0.5)
        methods_count = len(plan.get('methods', []))
        
        # 复杂度越高，置信度稍低；方法越多，置信度越高
        confidence = base_confidence + (methods_count * 0.05) - (complexity * 0.1)
        return max(0.3, min(0.95, confidence))
    
    def generate_recommendations(self, analysis_type: str) -> list:
        """生成建议"""
        if analysis_type == 'data_analysis':
            return [
                "建议增加数据样本量以提高分析精度",
                "推荐使用更高级的统计方法验证结果",
                "需要对异常值进行深入分析"
            ]
        elif analysis_type == 'pattern_analysis':
            return [
                "建议建立预测模型利用发现的模式",
                "推荐持续监控模式变化",
                "考虑将模式分析结果应用于决策优化"
            ]
        else:
            return [
                "建议进行更详细的专项分析",
                "推荐收集更多相关数据"
            ]
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察分析结果"""
        confidence = action_result.content.get('confidence', 0.5)
        findings_count = len(action_result.content.get('findings', []))
        
        observation = {
            'success_score': confidence,
            'task_completed': confidence > 0.6 and findings_count > 0,
            'analysis_quality': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low',
            'next_action': 'refine_analysis' if confidence < 0.7 else 'complete'
        }
        
        return observation


class OptimizerAgent(BaseAgent):
    """优化器Agent - 专门处理系统优化任务"""
    
    def __init__(self, agent_id: str, communication: CommunicationProtocol):
        super().__init__(agent_id, AgentRole.OPTIMIZER, communication)
        self.optimization_strategies = [
            'parameter_tuning',
            'algorithm_selection',
            'resource_allocation',
            'performance_enhancement'
        ]
        
    async def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """优化器思考过程"""
        goal = context.get('goal', 'general_optimization')
        
        # 获取系统当前状态
        messages = context.get('messages', [])
        current_performance = self.extract_performance_info(messages)
        
        plan = {
            'action_type': 'optimization',
            'target_area': self.identify_optimization_target(goal),
            'strategy': self.select_optimization_strategy(current_performance),
            'expected_improvement': self.estimate_improvement_potential(current_performance)
        }
        
        return plan
    
    def extract_performance_info(self, messages: list) -> dict:
        """从消息中提取性能信息"""
        performance_info = {
            'current_score': 0.5,
            'bottlenecks': [],
            'improvement_areas': []
        }
        
        # 分析消息中的性能数据
        for msg in messages:
            if msg.get('topic') == 'evaluation_results':
                evaluations = msg.get('message', {}).get('evaluations', [])
                if evaluations:
                    avg_score = sum(e.get('overall_score', 0.5) for e in evaluations) / len(evaluations)
                    performance_info['current_score'] = avg_score
        
        return performance_info
    
    def identify_optimization_target(self, goal: str) -> str:
        """识别优化目标"""
        if 'performance' in goal.lower():
            return 'performance'
        elif 'efficiency' in goal.lower():
            return 'efficiency'  
        elif 'resource' in goal.lower():
            return 'resource_usage'
        else:
            return 'general'
    
    def select_optimization_strategy(self, performance_info: dict) -> str:
        """选择优化策略"""
        current_score = performance_info.get('current_score', 0.5)
        
        if current_score < 0.4:
            return 'parameter_tuning'  # 大幅调整
        elif current_score < 0.7:
            return 'algorithm_selection'  # 算法优化
        else:
            return 'performance_enhancement'  # 精细优化
    
    def estimate_improvement_potential(self, performance_info: dict) -> float:
        """估算改进潜力"""
        current_score = performance_info.get('current_score', 0.5)
        return min(0.3, (1.0 - current_score) * 0.5)  # 最多30%的改进
    
    async def act(self, plan: Dict[str, Any]) -> AgentAction:
        """执行优化"""
        start_time = time.time()
        
        strategy = plan.get('strategy', 'parameter_tuning')
        target_area = plan.get('target_area', 'general')
        expected_improvement = plan.get('expected_improvement', 0.1)
        
        # 模拟优化过程
        optimization_result = {
            'strategy_used': strategy,
            'target_area': target_area,
            'optimizations': self.generate_optimizations(strategy, target_area),
            'expected_improvement': expected_improvement,
            'implementation_steps': self.generate_implementation_steps(strategy)
        }
        
        # 广播优化建议
        self.communication.publish(
            'optimization_proposal',
            optimization_result,
            self.agent_id
        )
        
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=ActionType.EXECUTE,
            content=optimization_result,
            metadata={
                'execution_time': time.time() - start_time,
                'output_score': expected_improvement,
                'optimization_scope': target_area
            }
        )
        
        self.action_history.append(action)
        return action
    
    def generate_optimizations(self, strategy: str, target_area: str) -> list:
        """生成具体优化方案"""
        optimizations = []
        
        if strategy == 'parameter_tuning':
            optimizations = [
                f"调整{target_area}相关参数，提高响应速度",
                "优化内存使用模式，减少资源消耗",
                "调整并发处理参数，提升吞吐量"
            ]
        elif strategy == 'algorithm_selection':
            optimizations = [
                f"为{target_area}选择更适合的算法",
                "实施算法组合策略，提升整体效果",
                "引入自适应算法选择机制"
            ]
        elif strategy == 'performance_enhancement':
            optimizations = [
                f"对{target_area}进行微调优化",
                "实施缓存策略，减少重复计算",
                "优化数据流处理路径"
            ]
        
        return optimizations
    
    def generate_implementation_steps(self, strategy: str) -> list:
        """生成实施步骤"""
        if strategy == 'parameter_tuning':
            return [
                "1. 备份当前参数配置",
                "2. 渐进式调整关键参数",
                "3. 监控性能变化",
                "4. 验证优化效果",
                "5. 固化最优配置"
            ]
        elif strategy == 'algorithm_selection':
            return [
                "1. 评估当前算法性能",
                "2. 候选算法性能测试",
                "3. 选择最优算法组合",
                "4. 平滑切换实施",
                "5. 效果验证与调优"
            ]
        else:
            return [
                "1. 性能基线建立",
                "2. 优化点识别",
                "3. 优化方案实施",
                "4. 效果评估",
                "5. 持续监控"
            ]
    
    async def observe(self, action_result: AgentAction) -> Dict[str, Any]:
        """观察优化结果"""
        expected_improvement = action_result.content.get('expected_improvement', 0.1)
        optimizations_count = len(action_result.content.get('optimizations', []))
        
        observation = {
            'success_score': min(0.9, 0.5 + expected_improvement * 2),
            'task_completed': optimizations_count > 0,
            'optimization_potential': 'high' if expected_improvement > 0.2 else 'medium',
            'next_action': 'implement' if expected_improvement > 0.1 else 'refine'
        }
        
        return observation


async def custom_agent_demo():
    """自定义Agent演示"""
    print("🤖 自定义Agent演示 - 数据分析师 & 优化器")
    print("=" * 60)
    
    # 创建系统
    system = AutonomousEvolutionarySystem()
    
    # 创建自定义Agent
    data_analyst = DataAnalystAgent("data_analyst_001", system.communication)
    optimizer = OptimizerAgent("optimizer_001", system.communication)
    
    # 添加到系统
    system.add_agent(data_analyst)
    system.add_agent(optimizer)
    
    print(f"\n✅ 创建了 {len(system.agents)} 个自定义Agent:")
    for agent_id, agent in system.agents.items():
        print(f"   - {agent_id}: {agent.__class__.__name__} ({agent.role.value})")
    
    # 运行数据分析任务
    print("\n🔍 任务1: 数据分析")
    result1 = await system.run_collaborative_task(
        goal="对用户行为数据进行pattern analysis，识别关键趋势",
        max_cycles=2
    )
    
    print(f"   分析结果: {result1['total_actions']} 个行动")
    if result1['final_metrics']:
        print(f"   性能得分: {result1['final_metrics'].composite_score:.3f}")
    
    # 运行优化任务
    print("\n⚡ 任务2: 系统优化")
    result2 = await system.run_collaborative_task(
        goal="优化系统performance，提升整体效率",
        max_cycles=2
    )
    
    print(f"   优化结果: {result2['total_actions']} 个行动")
    if result2['final_metrics']:
        print(f"   性能得分: {result2['final_metrics'].composite_score:.3f}")
    
    # 显示Agent进化状态
    print(f"\n🧠 Agent进化状态:")
    for agent_id, agent in system.agents.items():
        print(f"   {agent_id}:")
        print(f"     - 优化次数: {agent.optimization_counter}")
        print(f"     - 当前温度: {agent.temperature:.3f}")
        print(f"     - 记忆条目: {len(agent.memory)}")
        print(f"     - 成功模式: {len(agent.success_patterns)}")
    
    print("\n🎉 自定义Agent演示完成！")


if __name__ == "__main__":
    print("🎯 自定义Agent示例")
    print("展示如何创建专业化的自定义Agent并集成到系统中\n")
    
    asyncio.run(custom_agent_demo())
    
    print("\n" + "=" * 60)
    print("💡 学习要点:")
    print("1. 继承BaseAgent类创建自定义Agent")
    print("2. 实现think(), act(), observe()方法")
    print("3. 定义专业化的行为逻辑")
    print("4. 使用通信协议进行Agent间协作")
    print("5. 通过系统评估监控Agent性能")
    print("=" * 60)