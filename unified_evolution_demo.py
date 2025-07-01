"""
统一进化系统演示程序
展示整合优化后的完整进化系统功能
"""

import asyncio
import time
import random
import json
import logging
from typing import Dict, List, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入统一进化系统
try:
    from unified_evolution_system import UnifiedEvolutionSystem, EnhancedMemory, EvolutionStrategy
    logger.info("成功导入统一进化系统")
except ImportError as e:
    logger.error(f"导入统一进化系统失败: {e}")
    exit(1)

class EvolutionDemo:
    """进化系统演示类"""
    
    def __init__(self):
        self.evolution_system = UnifiedEvolutionSystem()
        self.demo_results = []
        logger.info("进化演示系统初始化完成")
    
    def demo_memory_management(self):
        """演示记忆管理功能"""
        logger.info("=== 记忆管理演示 ===")
        
        # 添加测试记忆
        test_memories = [
            EnhancedMemory(
                content="用户询问关于机器学习算法的问题",
                memory_type="conversation",
                importance=0.8,
                emotional_valence=0.3,
                tags=["机器学习", "算法", "问答"]
            ),
            EnhancedMemory(
                content="成功解决了一个复杂的编程问题",
                memory_type="experience", 
                importance=0.9,
                emotional_valence=0.8,
                tags=["编程", "问题解决", "成功"]
            ),
            EnhancedMemory(
                content="学习了新的深度学习技术",
                memory_type="knowledge",
                importance=0.7,
                emotional_valence=0.5,
                tags=["深度学习", "技术", "学习"]
            )
        ]
        
        added_memories = []
        for memory in test_memories:
            memory_id = self.evolution_system.add_enhanced_memory(memory)
            added_memories.append(memory_id)
            logger.info(f"添加记忆 ID: {memory_id}")
        
        # 搜索记忆
        logger.info("搜索'机器学习'相关记忆:")
        search_results = self.evolution_system.search_enhanced_memories("机器学习")
        for memory in search_results:
            logger.info(f"  - {memory.content[:50]}... (重要性: {memory.importance:.3f})")
        
        # 记忆巩固
        logger.info("执行记忆巩固...")
        self.evolution_system.consolidate_memories()
        
        # 应用遗忘曲线
        logger.info("应用遗忘曲线...")
        self.evolution_system.apply_forgetting_curve()
        
        return {
            'added_memories': len(added_memories),
            'search_results': len(search_results),
            'consolidation_completed': True
        }
    
    def demo_genetic_algorithm(self):
        """演示遗传算法"""
        logger.info("=== 遗传算法演示 ===")
        
        # 初始化种群
        self.evolution_system.initialize_population(size=20)
        logger.info(f"初始化种群大小: {len(self.evolution_system.population)}")
        
        # 进化多代
        generations = 5
        evolution_stats = []
        
        for gen in range(generations):
            stats = self.evolution_system.evolve_generation()
            evolution_stats.append(stats)
            logger.info(f"第{gen+1}代 - 最佳适应度: {stats['best_fitness']:.4f}, "
                       f"平均适应度: {stats['average_fitness']:.4f}")
        
        return {
            'generations_evolved': generations,
            'final_best_fitness': evolution_stats[-1]['best_fitness'],
            'final_average_fitness': evolution_stats[-1]['average_fitness'],
            'diversity_trend': [s['diversity'] for s in evolution_stats]
        }
    
    def demo_particle_swarm(self):
        """演示粒子群算法"""
        logger.info("=== 粒子群算法演示 ===")
        
        # 初始化粒子群
        self.evolution_system.initialize_swarm(dimensions=8)
        logger.info(f"初始化粒子群大小: {len(self.evolution_system.particles)}")
        
        # 更新多次
        iterations = 10
        swarm_stats = []
        
        for iteration in range(iterations):
            stats = self.evolution_system.update_particle_swarm()
            swarm_stats.append(stats)
            logger.info(f"迭代{iteration+1} - 最佳适应度: {stats['best_fitness']:.4f}, "
                       f"群体多样性: {stats['swarm_diversity']:.4f}")
        
        return {
            'iterations_completed': iterations,
            'final_best_fitness': swarm_stats[-1]['best_fitness'],
            'final_diversity': swarm_stats[-1]['swarm_diversity'],
            'convergence_trend': [s['best_fitness'] for s in swarm_stats]
        }
    
    def demo_adaptive_evolution(self):
        """演示自适应进化"""
        logger.info("=== 自适应进化演示 ===")
        
        # 模拟性能数据
        for i in range(50):
            performance_score = random.uniform(0.3, 0.9) + random.gauss(0, 0.1)
            performance_score = max(0, min(1, performance_score))
            self.evolution_system.performance_window.append({
                'score': performance_score,
                'timestamp': time.time() - (50-i) * 3600  # 模拟历史数据
            })
        
        logger.info(f"加载了{len(self.evolution_system.performance_window)}条性能数据")
        
        # 执行多次自适应进化
        evolution_cycles = 3
        adaptive_results = []
        
        for cycle in range(evolution_cycles):
            logger.info(f"执行第{cycle+1}次自适应进化...")
            
            # 选择策略
            strategy = self.evolution_system.select_evolution_strategy()
            logger.info(f"选择的策略: {strategy.value}")
            
            # 执行进化
            evolution_result = self.evolution_system.execute_unified_evolution()
            adaptive_results.append(evolution_result)
            
            logger.info(f"进化完成，执行时间: {evolution_result['execution_time']:.3f}秒")
        
        return {
            'evolution_cycles': evolution_cycles,
            'strategies_used': [r['strategy'] for r in adaptive_results],
            'total_execution_time': sum(r['execution_time'] for r in adaptive_results),
            'strategy_weights': {k.value: v for k, v in self.evolution_system.strategy_weights.items()}
        }
    
    def demo_system_integration(self):
        """演示系统集成功能"""
        logger.info("=== 系统集成演示 ===")
        
        # 获取系统状态
        status = self.evolution_system.get_system_status()
        logger.info("系统状态:")
        for key, value in status.items():
            logger.info(f"  {key}: {value}")
        
        # 综合性能测试
        start_time = time.time()
        
        # 同时运行多个组件
        memory_result = self.demo_memory_management()
        genetic_result = self.demo_genetic_algorithm()
        swarm_result = self.demo_particle_swarm()
        adaptive_result = self.demo_adaptive_evolution()
        
        total_time = time.time() - start_time
        
        # 保存演示结果
        demo_summary = {
            'timestamp': time.time(),
            'total_execution_time': total_time,
            'system_status': status,
            'memory_demo': memory_result,
            'genetic_demo': genetic_result,
            'swarm_demo': swarm_result,
            'adaptive_demo': adaptive_result,
            'integration_successful': True
        }
        
        # 保存到文件
        with open('unified_evolution_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(demo_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"演示完成，总耗时: {total_time:.2f}秒")
        logger.info("结果已保存到 unified_evolution_demo_results.json")
        
        return demo_summary
    
    def run_comprehensive_demo(self):
        """运行综合演示"""
        logger.info("🚀 开始统一进化系统综合演示")
        logger.info("="*60)
        
        try:
            # 运行系统集成演示
            results = self.demo_system_integration()
            
            # 打印总结
            logger.info("="*60)
            logger.info("📊 演示总结:")
            logger.info(f"✅ 记忆管理: 添加{results['memory_demo']['added_memories']}条记忆")
            logger.info(f"✅ 遗传算法: 进化{results['genetic_demo']['generations_evolved']}代")
            logger.info(f"✅ 粒子群: 完成{results['swarm_demo']['iterations_completed']}次迭代")
            logger.info(f"✅ 自适应进化: 执行{results['adaptive_demo']['evolution_cycles']}个周期")
            logger.info(f"⏱️  总耗时: {results['total_execution_time']:.2f}秒")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"演示过程中发生错误: {e}")
            return False

def main():
    """主函数"""
    try:
        # 创建演示实例
        demo = EvolutionDemo()
        
        # 运行综合演示
        success = demo.run_comprehensive_demo()
        
        if success:
            logger.info("🎉 统一进化系统演示成功完成！")
        else:
            logger.error("❌ 演示失败")
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()