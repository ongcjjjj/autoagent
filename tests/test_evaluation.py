#!/usr/bin/env python3
"""
评估系统测试模块 - 自主进化Agent系统
测试9维度评估系统的功能
"""

import unittest
import asyncio
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    AutonomousEvolutionarySystem,
    PerformanceMetrics,
    TrainingFreeEvaluator
)


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标类"""
    
    def setUp(self):
        """设置测试环境"""
        self.metrics = PerformanceMetrics(
            trainability=0.8,
            generalization=0.7,
            expressiveness=0.6,
            creativity_score=0.9,
            adaptation_rate=0.75,
            collaboration_efficiency=0.85,
            error_recovery_rate=0.8,
            knowledge_retention=0.7,
            innovation_index=0.65
        )
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        self.assertEqual(self.metrics.trainability, 0.8)
        self.assertEqual(self.metrics.generalization, 0.7)
        self.assertEqual(self.metrics.creativity_score, 0.9)
        self.assertGreater(self.metrics.composite_score, 0)
        self.assertLessEqual(self.metrics.composite_score, 1)
    
    def test_composite_score_calculation(self):
        """测试综合得分计算"""
        # 综合得分应该是加权平均
        expected_score = (
            0.8 * 0.15 +  # trainability
            0.7 * 0.15 +  # generalization
            0.6 * 0.10 +  # expressiveness
            0.9 * 0.15 +  # creativity_score
            0.75 * 0.10 + # adaptation_rate
            0.85 * 0.15 + # collaboration_efficiency
            0.8 * 0.10 +  # error_recovery_rate
            0.7 * 0.05 +  # knowledge_retention
            0.65 * 0.05   # innovation_index
        )
        
        self.assertAlmostEqual(self.metrics.composite_score, expected_score, places=3)
    
    def test_to_dict(self):
        """测试转换为字典"""
        metrics_dict = self.metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('trainability', metrics_dict)
        self.assertIn('composite_score', metrics_dict)
        self.assertEqual(len(metrics_dict), 10)  # 9个维度 + 综合得分
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'trainability': 0.8,
            'generalization': 0.7,
            'expressiveness': 0.6,
            'creativity_score': 0.9,
            'adaptation_rate': 0.75,
            'collaboration_efficiency': 0.85,
            'error_recovery_rate': 0.8,
            'knowledge_retention': 0.7,
            'innovation_index': 0.65
        }
        
        new_metrics = PerformanceMetrics.from_dict(data)
        
        self.assertEqual(new_metrics.trainability, 0.8)
        self.assertEqual(new_metrics.creativity_score, 0.9)
    
    def test_metrics_bounds(self):
        """测试指标边界值"""
        # 测试最小值
        min_metrics = PerformanceMetrics(
            trainability=0.0,
            generalization=0.0,
            expressiveness=0.0,
            creativity_score=0.0,
            adaptation_rate=0.0,
            collaboration_efficiency=0.0,
            error_recovery_rate=0.0,
            knowledge_retention=0.0,
            innovation_index=0.0
        )
        
        self.assertEqual(min_metrics.composite_score, 0.0)
        
        # 测试最大值
        max_metrics = PerformanceMetrics(
            trainability=1.0,
            generalization=1.0,
            expressiveness=1.0,
            creativity_score=1.0,
            adaptation_rate=1.0,
            collaboration_efficiency=1.0,
            error_recovery_rate=1.0,
            knowledge_retention=1.0,
            innovation_index=1.0
        )
        
        self.assertEqual(max_metrics.composite_score, 1.0)


class TestTrainingFreeEvaluator(unittest.TestCase):
    """测试训练无关评估器"""
    
    def setUp(self):
        """设置测试环境"""
        self.evaluator = TrainingFreeEvaluator()
    
    def test_evaluator_initialization(self):
        """测试评估器初始化"""
        self.assertIsInstance(self.evaluator, TrainingFreeEvaluator)
    
    def test_evaluate_trainability(self):
        """测试可训练性评估"""
        # 模拟模型参数
        model_params = {
            'param_count': 1000000,
            'layer_count': 10,
            'gradient_stats': {'mean': 0.01, 'std': 0.1}
        }
        
        trainability = self.evaluator.evaluate_trainability(model_params)
        
        self.assertIsInstance(trainability, float)
        self.assertGreaterEqual(trainability, 0)
        self.assertLessEqual(trainability, 1)
    
    def test_evaluate_generalization(self):
        """测试泛化能力评估"""
        model_complexity = 0.5
        
        generalization = self.evaluator.evaluate_generalization(model_complexity)
        
        self.assertIsInstance(generalization, float)
        self.assertGreaterEqual(generalization, 0)
        self.assertLessEqual(generalization, 1)
    
    def test_evaluate_expressiveness(self):
        """测试表达能力评估"""
        architecture_info = {
            'depth': 5,
            'width': 100,
            'activation_diversity': 0.8,
            'connection_density': 0.6
        }
        
        expressiveness = self.evaluator.evaluate_expressiveness(architecture_info)
        
        self.assertIsInstance(expressiveness, float)
        self.assertGreaterEqual(expressiveness, 0)
        self.assertLessEqual(expressiveness, 1)
    
    def test_evaluate_creativity(self):
        """测试创造性评估"""
        output_samples = [
            "创新的解决方案A",
            "标准方法B", 
            "独特的想法C"
        ]
        
        creativity = self.evaluator.evaluate_creativity(output_samples)
        
        self.assertIsInstance(creativity, float)
        self.assertGreaterEqual(creativity, 0)
        self.assertLessEqual(creativity, 1)
    
    def test_evaluate_adaptation_rate(self):
        """测试适应速度评估"""
        performance_history = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        adaptation_rate = self.evaluator.evaluate_adaptation_rate(performance_history)
        
        self.assertIsInstance(adaptation_rate, float)
        self.assertGreaterEqual(adaptation_rate, 0)
        self.assertLessEqual(adaptation_rate, 1)


class TestSystemEvaluation(unittest.TestCase):
    """测试系统级评估"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = AutonomousEvolutionarySystem()
    
    async def test_system_performance_evaluation(self):
        """测试系统性能评估"""
        # 创建团队
        team = self.system.create_standard_team()
        
        # 评估系统性能
        metrics = await self.system.evaluate_system_performance()
        
        # 检查评估结果
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertGreaterEqual(metrics.composite_score, 0)
        self.assertLessEqual(metrics.composite_score, 1)
        
        # 检查所有维度
        self.assertGreaterEqual(metrics.trainability, 0)
        self.assertGreaterEqual(metrics.generalization, 0)
        self.assertGreaterEqual(metrics.creativity_score, 0)
        self.assertGreaterEqual(metrics.collaboration_efficiency, 0)
    
    async def test_evaluation_consistency(self):
        """测试评估一致性"""
        # 创建团队
        team = self.system.create_standard_team()
        
        # 多次评估
        metrics1 = await self.system.evaluate_system_performance()
        metrics2 = await self.system.evaluate_system_performance()
        
        # 评估结果应该相对稳定
        score_diff = abs(metrics1.composite_score - metrics2.composite_score)
        self.assertLess(score_diff, 0.2)  # 差异不应超过0.2
    
    async def test_evaluation_after_task(self):
        """测试任务后评估"""
        # 创建团队
        team = self.system.create_standard_team()
        
        # 评估初始性能
        initial_metrics = await self.system.evaluate_system_performance()
        
        # 运行任务
        await self.system.run_collaborative_task(
            goal="测试任务",
            max_cycles=1
        )
        
        # 评估任务后性能
        final_metrics = await self.system.evaluate_system_performance()
        
        # 性能应该有所变化（学习效果）
        self.assertIsInstance(initial_metrics, PerformanceMetrics)
        self.assertIsInstance(final_metrics, PerformanceMetrics)


class TestEvaluationMetrics(unittest.TestCase):
    """测试评估指标计算"""
    
    def test_collaboration_efficiency_calculation(self):
        """测试协作效率计算"""
        evaluator = TrainingFreeEvaluator()
        
        # 模拟协作数据
        collaboration_data = {
            'message_count': 50,
            'task_completion_rate': 0.8,
            'conflict_resolution_time': 10.0,
            'resource_utilization': 0.75
        }
        
        efficiency = evaluator.evaluate_collaboration_efficiency(collaboration_data)
        
        self.assertIsInstance(efficiency, float)
        self.assertGreaterEqual(efficiency, 0)
        self.assertLessEqual(efficiency, 1)
    
    def test_error_recovery_calculation(self):
        """测试错误恢复率计算"""
        evaluator = TrainingFreeEvaluator()
        
        # 模拟错误恢复数据
        error_data = {
            'total_errors': 10,
            'recovered_errors': 8,
            'recovery_time': [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0],
            'performance_degradation': 0.1
        }
        
        recovery_rate = evaluator.evaluate_error_recovery_rate(error_data)
        
        self.assertIsInstance(recovery_rate, float)
        self.assertGreaterEqual(recovery_rate, 0)
        self.assertLessEqual(recovery_rate, 1)
    
    def test_knowledge_retention_calculation(self):
        """测试知识保持率计算"""
        evaluator = TrainingFreeEvaluator()
        
        # 模拟知识保持数据
        knowledge_data = {
            'learned_patterns': 20,
            'retained_patterns': 18,
            'memory_usage': 0.8,
            'pattern_accuracy': 0.9
        }
        
        retention_rate = evaluator.evaluate_knowledge_retention(knowledge_data)
        
        self.assertIsInstance(retention_rate, float)
        self.assertGreaterEqual(retention_rate, 0)
        self.assertLessEqual(retention_rate, 1)
    
    def test_innovation_index_calculation(self):
        """测试创新指数计算"""
        evaluator = TrainingFreeEvaluator()
        
        # 模拟创新数据
        innovation_data = {
            'novel_solutions': 5,
            'total_solutions': 20,
            'solution_diversity': 0.7,
            'breakthrough_count': 2
        }
        
        innovation_index = evaluator.evaluate_innovation_index(innovation_data)
        
        self.assertIsInstance(innovation_index, float)
        self.assertGreaterEqual(innovation_index, 0)
        self.assertLessEqual(innovation_index, 1)


def run_async_test(test_func):
    """运行异步测试的辅助函数"""
    def wrapper(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_func(self))
        finally:
            loop.close()
    return wrapper


# 包装异步测试
TestSystemEvaluation.test_system_performance_evaluation = run_async_test(
    TestSystemEvaluation.test_system_performance_evaluation
)
TestSystemEvaluation.test_evaluation_consistency = run_async_test(
    TestSystemEvaluation.test_evaluation_consistency
)
TestSystemEvaluation.test_evaluation_after_task = run_async_test(
    TestSystemEvaluation.test_evaluation_after_task
)


if __name__ == '__main__':
    print("🧪 运行评估系统测试套件")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)