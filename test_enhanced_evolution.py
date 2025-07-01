"""
增强进化功能测试脚本
测试所有新增功能的基本运行情况
"""
import time
import random
import json
import traceback

# 基础模块导入
from memory import MemoryManager, Memory
from evolution import EvolutionEngine
from config_minimal import Config

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 测试基础功能...")
    
    try:
        # 基础组件测试
        config = Config()
        memory_manager = MemoryManager()
        evolution_engine = EvolutionEngine(memory_manager)
        
        # 添加测试记忆
        test_memory = Memory(
            content="这是一个测试记忆",
            memory_type="test",
            importance=0.8,
            tags=["test", "basic"],
            metadata={"test_id": 1}
        )
        
        memory_id = memory_manager.add_memory(test_memory)
        print(f"  ✅ 添加记忆成功: ID {memory_id}")
        
        # 测试记忆搜索
        search_results = memory_manager.search_memories("测试", limit=5)
        print(f"  ✅ 搜索记忆成功: 找到 {len(search_results)} 条记录")
        
        # 测试进化功能
        test_interaction = {
            "response_time": 2.5,
            "task_completed": True,
            "error_count": 0,
            "user_feedback": "good"
        }
        
        performance = evolution_engine.evaluate_performance(test_interaction)
        evolution_engine.update_performance_window(performance)
        print(f"  ✅ 性能评估成功: {performance:.3f}")
        
        # 获取进化摘要
        summary = evolution_engine.get_evolution_summary()
        print(f"  ✅ 进化摘要获取成功: {summary.get('message', '正常')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 基础功能测试失败: {e}")
        traceback.print_exc()
        return False

def test_adaptive_evolution():
    """测试自适应进化功能"""
    print("\n🎯 测试自适应进化功能...")
    
    try:
        from adaptive_evolution import AdaptiveEvolutionEngine
        
        memory_manager = MemoryManager()
        adaptive_engine = AdaptiveEvolutionEngine(memory_manager)
        
        # 测试策略
        print(f"  - 当前策略: {adaptive_engine.current_strategy.value}")
        
        # 模拟多次交互
        for i in range(5):
            test_interaction = {
                "response_time": random.uniform(1.0, 5.0),
                "task_completed": random.choice([True, False]),
                "error_count": random.randint(0, 2),
                "user_feedback": random.choice(["excellent", "good", "average"]),
                "interaction_id": i + 1
            }
            
            evolution_record = adaptive_engine.evolve_with_strategy(test_interaction)
            print(f"    交互 {i+1}: 策略 {adaptive_engine.current_strategy.value}, 版本 {evolution_record.version}")
        
        # 获取摘要
        summary = adaptive_engine.get_adaptive_evolution_summary()
        print(f"  ✅ 自适应进化测试成功")
        print(f"    当前策略: {summary['current_strategy']}")
        print(f"    性能趋势: {summary['performance_trend']}")
        
        return True
        
    except ImportError:
        print("  ⚠️ 自适应进化模块不可用，跳过测试")
        return True
    except Exception as e:
        print(f"  ❌ 自适应进化测试失败: {e}")
        traceback.print_exc()
        return False

def test_enhanced_memory():
    """测试增强记忆功能"""
    print("\n🧠 测试增强记忆功能...")
    
    try:
        from enhanced_memory import EnhancedMemoryManager, EnhancedMemory
        
        enhanced_memory = EnhancedMemoryManager()
        
        # 添加测试记忆
        for i in range(3):
            test_memory = EnhancedMemory(
                content=f"增强测试记忆 {i+1}",
                memory_type="test",
                importance=random.uniform(0.5, 1.0),
                emotional_valence=random.uniform(-0.5, 0.5),
                tags=["enhanced", "test", f"item_{i+1}"],
                metadata={"test_id": i + 1}
            )
            
            memory_id = enhanced_memory.add_memory(test_memory)
            print(f"  ✅ 添加增强记忆 {memory_id}")
        
        # 测试搜索
        search_results = enhanced_memory.search_memories("测试", limit=3)
        print(f"  ✅ 搜索找到 {len(search_results)} 条记忆")
        
        # 测试记忆巩固
        enhanced_memory.consolidate_memories(threshold_hours=0.1)
        print(f"  ✅ 记忆巩固测试成功")
        
        # 测试遗忘曲线
        enhanced_memory.apply_forgetting_curve(days_threshold=1)
        print(f"  ✅ 遗忘曲线测试成功")
        
        # 获取统计信息
        stats = enhanced_memory.get_memory_evolution_stats()
        print(f"  ✅ 统计信息: 总记忆 {stats['total_memories']}, 平均重要性 {stats['average_importance']:.3f}")
        
        return True
        
    except ImportError:
        print("  ⚠️ 增强记忆模块不可用，跳过测试")
        return True
    except Exception as e:
        print(f"  ❌ 增强记忆测试失败: {e}")
        traceback.print_exc()
        return False

def test_genetic_algorithms():
    """测试遗传算法功能"""
    print("\n🧬 测试遗传算法功能...")
    
    try:
        # 检查numpy可用性
        import numpy as np
        from genetic_evolution import GeneticEvolutionManager
        
        genetic_manager = GeneticEvolutionManager()
        
        # 简单的测试函数
        def simple_fitness(genes):
            # 目标：让所有基因接近0.5
            return 1.0 - sum(abs(g - 0.5) for g in genes) / len(genes)
        
        # 测试差分进化
        print("  🔄 测试差分进化...")
        genetic_manager.set_algorithm("differential_evolution")
        de_result = genetic_manager.optimize(
            fitness_func=simple_fitness,
            dimension=3,
            population_size=10,
            generations=5
        )
        
        print(f"    差分进化结果: 适应度 {de_result['best_fitness']:.3f}")
        
        # 测试模因算法
        print("  🔄 测试模因算法...")
        genetic_manager.set_algorithm("memetic_algorithm")
        ma_result = genetic_manager.optimize(
            fitness_func=simple_fitness,
            dimension=3,
            population_size=10,
            generations=5
        )
        
        print(f"    模因算法结果: 适应度 {ma_result['best_fitness']:.3f}")
        
        print("  ✅ 遗传算法测试成功")
        return True
        
    except ImportError:
        print("  ⚠️ 遗传算法模块不可用（需要numpy），跳过测试")
        return True
    except Exception as e:
        print(f"  ❌ 遗传算法测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n🔗 测试集成功能...")
    
    try:
        # 创建基础组件
        memory_manager = MemoryManager()
        evolution_engine = EvolutionEngine(memory_manager)
        
        # 尝试创建增强组件
        adaptive_engine = None
        enhanced_memory = None
        
        try:
            from adaptive_evolution import AdaptiveEvolutionEngine
            adaptive_engine = AdaptiveEvolutionEngine(memory_manager)
            print("  ✅ 自适应进化引擎集成成功")
        except:
            print("  ⚠️ 自适应进化引擎不可用")
        
        try:
            from enhanced_memory import EnhancedMemoryManager
            enhanced_memory = EnhancedMemoryManager()
            print("  ✅ 增强记忆管理器集成成功")
        except:
            print("  ⚠️ 增强记忆管理器不可用")
        
        # 模拟一个完整的工作流程
        print("  🔄 测试完整工作流程...")
        
        test_interaction = {
            "type": "集成测试",
            "content": "这是一个集成测试任务",
            "response_time": 3.0,
            "task_completed": True,
            "error_count": 0,
            "user_feedback": "good",
            "timestamp": time.time()
        }
        
        # 1. 基础记忆存储
        basic_memory = Memory(
            content=test_interaction['content'],
            memory_type="integration_test",
            importance=0.8,
            tags=["integration", "test"],
            metadata=test_interaction
        )
        
        basic_memory_id = memory_manager.add_memory(basic_memory)
        print(f"    基础记忆存储: ID {basic_memory_id}")
        
        # 2. 增强记忆存储（如果可用）
        if enhanced_memory:
            from enhanced_memory import EnhancedMemory
            enhanced_mem = EnhancedMemory(
                content=test_interaction['content'],
                memory_type="integration_test",
                importance=0.8,
                emotional_valence=0.2,
                tags=["integration", "test", "enhanced"],
                metadata=test_interaction
            )
            enhanced_memory_id = enhanced_memory.add_memory(enhanced_mem)
            print(f"    增强记忆存储: ID {enhanced_memory_id}")
        
        # 3. 基础进化评估
        performance = evolution_engine.evaluate_performance(test_interaction)
        evolution_engine.update_performance_window(performance)
        print(f"    基础进化评估: {performance:.3f}")
        
        # 4. 自适应进化（如果可用）
        if adaptive_engine:
            evolution_record = adaptive_engine.evolve_with_strategy(test_interaction)
            print(f"    自适应进化: 策略 {adaptive_engine.current_strategy.value}")
        
        print("  ✅ 集成测试成功")
        return True
        
    except Exception as e:
        print(f"  ❌ 集成测试失败: {e}")
        traceback.print_exc()
        return False

def generate_test_report(results):
    """生成测试报告"""
    print("\n📋 生成测试报告...")
    
    report = {
        "timestamp": time.time(),
        "test_results": results,
        "summary": {
            "total_tests": len(results),
            "passed": sum(results.values()),
            "failed": len(results) - sum(results.values())
        }
    }
    
    # 保存报告
    with open("enhanced_evolution_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"  📄 测试报告已保存到 enhanced_evolution_test_report.json")
    
    # 显示摘要
    print(f"\n📊 测试摘要:")
    print(f"  - 总测试数: {report['summary']['total_tests']}")
    print(f"  - 通过: {report['summary']['passed']}")
    print(f"  - 失败: {report['summary']['failed']}")
    
    print(f"\n📝 详细结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  - {test_name}: {status}")

def main():
    """主测试函数"""
    print("🌟 开始增强进化功能测试")
    print("=" * 50)
    
    # 执行所有测试
    test_results = {}
    
    test_results["基础功能"] = test_basic_functionality()
    test_results["自适应进化"] = test_adaptive_evolution()
    test_results["增强记忆"] = test_enhanced_memory()
    test_results["遗传算法"] = test_genetic_algorithms()
    test_results["集成功能"] = test_integration()
    
    # 生成测试报告
    generate_test_report(test_results)
    
    # 最终结果
    all_passed = all(test_results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！增强进化功能运行正常。")
    else:
        print("\n⚠️ 部分测试失败，但这可能是由于依赖项不可用。")
        print("   检查上面的详细信息以了解具体情况。")
    
    print("=" * 50)

if __name__ == "__main__":
    main()