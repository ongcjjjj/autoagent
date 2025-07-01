"""
增强自主进化Agent演示程序
展示遗传进化、增强记忆、自适应进化等所有新功能
"""
import asyncio
import time
import random
import json
from typing import Dict, List, Any

# 导入基础模块
from memory import MemoryManager, Memory
from evolution import EvolutionEngine, EvolutionMetrics
from openai_client import OpenAIClient
from config_minimal import Config

# 导入增强模块（处理可能的导入错误）
try:
    from adaptive_evolution import AdaptiveEvolutionEngine, EvolutionStrategy
    ADAPTIVE_AVAILABLE = True
except ImportError:
    print("自适应进化模块不可用，将使用基础进化功能")
    ADAPTIVE_AVAILABLE = False

try:
    from enhanced_memory import EnhancedMemoryManager, EnhancedMemory
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError:
    print("增强记忆模块不可用，将使用基础记忆功能")
    ENHANCED_MEMORY_AVAILABLE = False

try:
    # 尝试导入有数值计算依赖的模块
    import numpy as np
    from genetic_evolution import GeneticEvolutionManager
    GENETIC_AVAILABLE = True
except ImportError:
    print("遗传进化模块不可用（需要numpy），将跳过相关功能")
    GENETIC_AVAILABLE = False

class EnhancedAgentDemo:
    """增强Agent演示类"""
    
    def __init__(self):
        # 基础组件
        self.config = Config()
        self.memory_manager = MemoryManager()
        self.evolution_engine = EvolutionEngine(self.memory_manager)
        self.openai_client = None
        
        # 增强组件
        self.adaptive_engine = None
        self.enhanced_memory = None
        self.genetic_manager = None
        
        # 演示数据
        self.demo_interactions = []
        self.performance_log = []
        
        self.initialize_components()
    
    def initialize_components(self):
        """初始化所有组件"""
        print("🚀 初始化增强自主进化Agent...")
        
        # 初始化OpenAI客户端
        try:
            self.openai_client = OpenAIClient(self.config)
            print("✅ OpenAI客户端初始化成功")
        except Exception as e:
            print(f"⚠️ OpenAI客户端初始化失败: {e}")
        
        # 初始化自适应进化引擎
        if ADAPTIVE_AVAILABLE:
            try:
                self.adaptive_engine = AdaptiveEvolutionEngine(self.memory_manager)
                print("✅ 自适应进化引擎初始化成功")
            except Exception as e:
                print(f"⚠️ 自适应进化引擎初始化失败: {e}")
        
        # 初始化增强记忆管理器
        if ENHANCED_MEMORY_AVAILABLE:
            try:
                self.enhanced_memory = EnhancedMemoryManager()
                print("✅ 增强记忆管理器初始化成功")
            except Exception as e:
                print(f"⚠️ 增强记忆管理器初始化失败: {e}")
        
        # 初始化遗传进化管理器
        if GENETIC_AVAILABLE:
            try:
                self.genetic_manager = GeneticEvolutionManager()
                print("✅ 遗传进化管理器初始化成功")
            except Exception as e:
                print(f"⚠️ 遗传进化管理器初始化失败: {e}")
        
        print("🎯 Agent初始化完成！")
    
    def generate_demo_interactions(self, count: int = 20):
        """生成演示交互数据"""
        print(f"\n📊 生成 {count} 个演示交互...")
        
        interaction_types = [
            "问答", "任务执行", "代码生成", "文本分析", "创意写作"
        ]
        
        for i in range(count):
            interaction_type = random.choice(interaction_types)
            
            # 模拟真实的交互数据
            interaction = {
                "id": i + 1,
                "type": interaction_type,
                "timestamp": time.time() - random.randint(0, 3600 * 24),  # 过去24小时内
                "response_time": random.uniform(0.5, 8.0),
                "task_completed": random.choice([True, True, True, False]),  # 75%成功率
                "error_count": random.randint(0, 3),
                "user_feedback": random.choice([
                    "excellent", "good", "good", "average", "poor"
                ]),
                "content": f"用户询问关于{interaction_type}的问题 #{i+1}",
                "complexity": random.uniform(0.1, 1.0)
            }
            
            self.demo_interactions.append(interaction)
        
        print("✅ 演示交互数据生成完成")
    
    def demonstrate_basic_evolution(self):
        """演示基础进化功能"""
        print("\n🧬 演示基础进化功能...")
        
        for i, interaction in enumerate(self.demo_interactions[:10]):
            print(f"\n处理交互 {i+1}: {interaction['type']}")
            
            # 评估性能
            performance = self.evolution_engine.evaluate_performance(interaction)
            self.evolution_engine.update_performance_window(performance)
            
            # 添加记忆
            memory = Memory(
                content=interaction['content'],
                memory_type="conversation",
                importance=performance * 0.8 + 0.2,
                tags=[interaction['type'], "demo"],
                metadata=interaction
            )
            self.memory_manager.add_memory(memory)
            
            print(f"  - 性能评分: {performance:.3f}")
            
            # 检查是否需要进化
            if self.evolution_engine.should_evolve():
                print("  🔄 触发进化...")
                evolution_record = self.evolution_engine.execute_evolution()
                print(f"  ✨ 进化完成: {evolution_record.version}")
                print(f"  📈 改进领域: {', '.join(evolution_record.improvement_areas)}")
        
        # 显示进化摘要
        summary = self.evolution_engine.get_evolution_summary()
        print(f"\n📊 基础进化摘要:")
        print(f"  - 总进化次数: {summary['total_evolutions']}")
        print(f"  - 性能趋势: {summary['performance_trend']}")
    
    def demonstrate_adaptive_evolution(self):
        """演示自适应进化功能"""
        if not self.adaptive_engine:
            print("\n⚠️ 自适应进化功能不可用")
            return
        
        print("\n🎯 演示自适应进化功能...")
        
        for i, interaction in enumerate(self.demo_interactions[10:]):
            print(f"\n处理自适应交互 {i+1}: {interaction['type']}")
            
            # 使用自适应进化
            evolution_record = self.adaptive_engine.evolve_with_strategy(interaction)
            
            print(f"  - 当前策略: {self.adaptive_engine.current_strategy.value}")
            print(f"  - 进化版本: {evolution_record.version}")
            
            # 记录性能
            self.performance_log.append({
                "interaction_id": i + 11,
                "strategy": self.adaptive_engine.current_strategy.value,
                "performance": evolution_record.metrics.success_rate,
                "timestamp": time.time()
            })
        
        # 显示自适应进化摘要
        summary = self.adaptive_engine.get_adaptive_evolution_summary()
        print(f"\n📊 自适应进化摘要:")
        print(f"  - 当前策略: {summary['current_strategy']}")
        print(f"  - 性能趋势: {summary['performance_trend']}")
        print(f"  - 帕累托前沿大小: {summary['multi_objective_analysis']['pareto_front_size']}")
        print(f"  - 环境适应次数: {summary['environment_adaptation']['total_adaptations']}")
    
    def demonstrate_enhanced_memory(self):
        """演示增强记忆功能"""
        if not self.enhanced_memory:
            print("\n⚠️ 增强记忆功能不可用")
            return
        
        print("\n🧠 演示增强记忆功能...")
        
        # 添加增强记忆
        for i, interaction in enumerate(self.demo_interactions[:5]):
            enhanced_mem = EnhancedMemory(
                content=interaction['content'],
                memory_type="experience",
                importance=random.uniform(0.3, 1.0),
                emotional_valence=random.uniform(-0.5, 0.5),
                tags=[interaction['type'], "enhanced", "demo"],
                metadata=interaction
            )
            
            memory_id = self.enhanced_memory.add_memory(enhanced_mem)
            print(f"  ✅ 添加增强记忆 {memory_id}: {interaction['type']}")
        
        # 演示智能搜索
        print("\n🔍 演示智能记忆搜索:")
        search_results = self.enhanced_memory.search_memories("问答", limit=3)
        for memory in search_results:
            print(f"  - 记忆 {memory.id}: {memory.content[:50]}...")
            print(f"    重要性: {memory.importance:.3f}, 访问次数: {memory.access_frequency}")
        
        # 演示记忆巩固
        print("\n🔄 执行记忆巩固...")
        self.enhanced_memory.consolidate_memories(threshold_hours=1.0)
        
        # 演示记忆聚类
        print("\n📊 执行记忆聚类...")
        self.enhanced_memory.cluster_memories(num_clusters=3)
        
        # 显示记忆统计
        stats = self.enhanced_memory.get_memory_evolution_stats()
        print(f"\n📈 增强记忆统计:")
        print(f"  - 总记忆数: {stats['total_memories']}")
        print(f"  - 平均重要性: {stats['average_importance']:.3f}")
        print(f"  - 图谱节点数: {stats['graph_stats']['nodes']}")
        print(f"  - 图谱边数: {stats['graph_stats']['edges']}")
        print(f"  - 社区数: {stats['graph_stats']['communities']}")
    
    def demonstrate_genetic_optimization(self):
        """演示遗传算法优化"""
        if not self.genetic_manager:
            print("\n⚠️ 遗传进化功能不可用")
            return
        
        print("\n🧬 演示遗传算法优化...")
        
        # 定义一个简单的优化问题
        def fitness_function(genes):
            """简单的适应度函数：最大化所有基因的平方和"""
            return sum(g * g for g in genes)
        
        # 演示差分进化
        print("\n🔄 差分进化算法优化:")
        self.genetic_manager.set_algorithm("differential_evolution")
        de_result = self.genetic_manager.optimize(
            fitness_func=fitness_function,
            dimension=5,
            population_size=20,
            generations=30
        )
        
        print(f"  ✅ 差分进化结果:")
        print(f"    最优解: {[f'{x:.3f}' for x in de_result['best_solution']]}")
        print(f"    最优适应度: {de_result['best_fitness']:.3f}")
        print(f"    最终种群多样性: {de_result['final_population_stats']['diversity']:.3f}")
        
        # 演示模因算法
        print("\n🔄 模因算法优化:")
        self.genetic_manager.set_algorithm("memetic_algorithm")
        ma_result = self.genetic_manager.optimize(
            fitness_func=fitness_function,
            dimension=5,
            population_size=20,
            generations=30
        )
        
        print(f"  ✅ 模因算法结果:")
        print(f"    最优解: {[f'{x:.3f}' for x in ma_result['best_solution']]}")
        print(f"    最优适应度: {ma_result['best_fitness']:.3f}")
        print(f"    最终种群多样性: {ma_result['final_population_stats']['diversity']:.3f}")
        
        # 演示协同进化
        print("\n🔄 协同进化算法优化:")
        self.genetic_manager.set_algorithm("coevolutionary")
        coevo_result = self.genetic_manager.optimize(
            fitness_func=fitness_function,
            dimension=5,
            population_size=15,  # 较小的种群用于协同进化
            generations=20
        )
        
        print(f"  ✅ 协同进化结果:")
        print(f"    最优解: {[f'{x:.3f}' for x in coevo_result['best_solution']]}")
        print(f"    最优适应度: {coevo_result['best_fitness']:.3f}")
        print(f"    物种数: {coevo_result['summary']['num_species']}")
        print(f"    最优物种: {coevo_result['summary']['best_species']}")
    
    def demonstrate_integrated_workflow(self):
        """演示完整的集成工作流程"""
        print("\n🔗 演示完整集成工作流程...")
        
        # 模拟一个复杂的学习任务
        complex_task = {
            "type": "复杂推理",
            "content": "解决多步骤逻辑推理问题",
            "difficulty": 0.8,
            "timestamp": time.time(),
            "response_time": 5.2,
            "task_completed": True,
            "error_count": 1,
            "user_feedback": "good"
        }
        
        print(f"📝 处理复杂任务: {complex_task['content']}")
        
        # 1. 基础记忆存储
        basic_memory = Memory(
            content=complex_task['content'],
            memory_type="experience",
            importance=0.9,
            tags=["complex", "reasoning"],
            metadata=complex_task
        )
        memory_id = self.memory_manager.add_memory(basic_memory)
        print(f"  💾 基础记忆存储: ID {memory_id}")
        
        # 2. 增强记忆存储（如果可用）
        if self.enhanced_memory:
            enhanced_mem = EnhancedMemory(
                content=complex_task['content'],
                memory_type="experience",
                importance=0.9,
                emotional_valence=0.3,  # 正面体验
                tags=["complex", "reasoning", "success"],
                metadata=complex_task
            )
            enhanced_id = self.enhanced_memory.add_memory(enhanced_mem)
            print(f"  🧠 增强记忆存储: ID {enhanced_id}")
        
        # 3. 基础进化评估
        performance = self.evolution_engine.evaluate_performance(complex_task)
        self.evolution_engine.update_performance_window(performance)
        print(f"  📊 性能评估: {performance:.3f}")
        
        # 4. 自适应进化（如果可用）
        if self.adaptive_engine:
            evolution_record = self.adaptive_engine.evolve_with_strategy(complex_task)
            print(f"  🎯 自适应进化: 策略 {self.adaptive_engine.current_strategy.value}")
            print(f"    进化版本: {evolution_record.version}")
        
        # 5. 相关记忆检索
        related_memories = self.memory_manager.search_memories("推理", limit=3)
        print(f"  🔍 检索到 {len(related_memories)} 个相关记忆")
        
        # 6. 增强记忆上下文（如果可用）
        if self.enhanced_memory and enhanced_id:
            context_memories = self.enhanced_memory.get_contextual_memories(enhanced_id, 3)
            print(f"  🌐 增强上下文: {len(context_memories)} 个相关记忆")
        
        # 7. 记忆巩固和遗忘
        if self.enhanced_memory:
            self.enhanced_memory.consolidate_memories()
            self.enhanced_memory.apply_forgetting_curve()
            print(f"  🔄 执行记忆巩固和遗忘曲线")
        
        print("  ✅ 集成工作流程完成")
    
    async def demonstrate_openai_integration(self):
        """演示OpenAI集成（如果可用）"""
        if not self.openai_client:
            print("\n⚠️ OpenAI集成不可用")
            return
        
        print("\n🤖 演示OpenAI集成...")
        
        try:
            # 测试连接
            is_connected = await self.openai_client.test_connection()
            if not is_connected:
                print("  ❌ OpenAI连接失败")
                return
            
            print("  ✅ OpenAI连接成功")
            
            # 示例对话
            messages = [
                {"role": "user", "content": "什么是自适应进化算法？"}
            ]
            
            response = await self.openai_client.chat_completion(messages)
            if response:
                print(f"  🎯 AI回答: {response[:100]}...")
                
                # 将对话存储到记忆中
                conversation_memory = Memory(
                    content=f"用户: 什么是自适应进化算法？\nAI: {response[:200]}...",
                    memory_type="conversation",
                    importance=0.7,
                    tags=["openai", "conversation", "algorithm"],
                    metadata={"model": self.config.model}
                )
                self.memory_manager.add_memory(conversation_memory)
                print("  💾 对话已存储到记忆中")
            
        except Exception as e:
            print(f"  ❌ OpenAI集成演示失败: {e}")
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n📋 生成综合演示报告...")
        
        report = {
            "timestamp": time.time(),
            "demo_summary": {
                "total_interactions": len(self.demo_interactions),
                "performance_logs": len(self.performance_log),
                "components_tested": []
            },
            "basic_evolution": {},
            "adaptive_evolution": {},
            "enhanced_memory": {},
            "genetic_optimization": {},
            "integration_status": {}
        }
        
        # 基础进化报告
        if hasattr(self, 'evolution_engine'):
            evo_summary = self.evolution_engine.get_evolution_summary()
            report["basic_evolution"] = evo_summary
            report["demo_summary"]["components_tested"].append("基础进化")
        
        # 自适应进化报告
        if self.adaptive_engine:
            adaptive_summary = self.adaptive_engine.get_adaptive_evolution_summary()
            report["adaptive_evolution"] = adaptive_summary
            report["demo_summary"]["components_tested"].append("自适应进化")
        
        # 增强记忆报告
        if self.enhanced_memory:
            memory_stats = self.enhanced_memory.get_memory_evolution_stats()
            report["enhanced_memory"] = memory_stats
            report["demo_summary"]["components_tested"].append("增强记忆")
        
        # 遗传优化报告
        if self.genetic_manager:
            genetic_history = self.genetic_manager.get_optimization_history()
            report["genetic_optimization"] = {
                "total_optimizations": len(genetic_history),
                "algorithms_used": list(set(opt["algorithm"] for opt in genetic_history))
            }
            report["demo_summary"]["components_tested"].append("遗传优化")
        
        # 集成状态
        report["integration_status"] = {
            "adaptive_evolution": ADAPTIVE_AVAILABLE,
            "enhanced_memory": ENHANCED_MEMORY_AVAILABLE,
            "genetic_algorithms": GENETIC_AVAILABLE,
            "openai_client": self.openai_client is not None
        }
        
        # 保存报告
        with open("enhanced_agent_demo_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("  📄 报告已保存到 enhanced_agent_demo_report.json")
        
        # 显示摘要
        print(f"\n📊 演示摘要:")
        print(f"  - 测试交互数: {report['demo_summary']['total_interactions']}")
        print(f"  - 组件测试: {', '.join(report['demo_summary']['components_tested'])}")
        print(f"  - 模块可用性:")
        for module, available in report["integration_status"].items():
            status = "✅" if available else "❌"
            print(f"    {status} {module}")
    
    async def run_complete_demo(self):
        """运行完整演示"""
        print("🌟 开始增强自主进化Agent完整演示")
        print("=" * 60)
        
        # 1. 生成演示数据
        self.generate_demo_interactions()
        
        # 2. 基础进化演示
        self.demonstrate_basic_evolution()
        
        # 3. 自适应进化演示
        self.demonstrate_adaptive_evolution()
        
        # 4. 增强记忆演示
        self.demonstrate_enhanced_memory()
        
        # 5. 遗传算法演示
        self.demonstrate_genetic_optimization()
        
        # 6. 集成工作流程演示
        self.demonstrate_integrated_workflow()
        
        # 7. OpenAI集成演示
        await self.demonstrate_openai_integration()
        
        # 8. 生成综合报告
        self.generate_comprehensive_report()
        
        print("\n🎉 演示完成！")
        print("=" * 60)

def main():
    """主函数"""
    demo = EnhancedAgentDemo()
    
    # 运行演示
    try:
        asyncio.run(demo.run_complete_demo())
    except KeyboardInterrupt:
        print("\n🛑 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()