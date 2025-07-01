"""
简化的自我进化Agent演示脚本
不依赖额外的库，避免类型错误
"""
import asyncio
import os
import sys
import time

# 添加当前目录到Python路径
sys.path.insert(0, '.')

try:
    from config import config
    from memory import MemoryManager, Memory
    from evolution import EvolutionEngine
    # 不导入openai_client，避免依赖问题
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有Python文件都在当前目录")
    sys.exit(1)

def simple_demo():
    """简化演示"""
    print("🚀 自我进化Agent核心功能演示")
    print("=" * 50)
    
    try:
        # 1. 配置管理演示
        print("📋 1. 配置管理演示")
        print(f"   默认模型: {config.openai_config.model}")
        print(f"   最大tokens: {config.openai_config.max_tokens}")
        print(f"   温度设置: {config.openai_config.temperature}")
        
        # 更新配置
        config.update_openai_config(model="gpt-4", temperature=0.8)
        print(f"   更新后模型: {config.openai_config.model}")
        print(f"   更新后温度: {config.openai_config.temperature}")
        print("   ✅ 配置管理正常\n")
        
        # 2. 记忆系统演示
        print("🧠 2. 记忆系统演示")
        memory_manager = MemoryManager("demo_memory.db")
        
        # 添加一些示例记忆
        demo_memories = [
            Memory(
                content="用户询问了Agent的自我进化能力",
                memory_type="conversation",
                importance=0.8,
                tags=["evolution", "capability"]
            ),
            Memory(
                content="Agent成功回答了关于配置的问题",
                memory_type="experience",
                importance=0.7,
                tags=["config", "success"]
            ),
            Memory(
                content="学习到用户喜欢详细的技术解释",
                memory_type="knowledge",
                importance=0.9,
                tags=["user_preference", "learning"]
            )
        ]
        
        for memory in demo_memories:
            memory_id = memory_manager.add_memory(memory)
            print(f"   添加记忆 ID {memory_id}: {memory.content[:30]}...")
        
        # 搜索记忆
        search_results = memory_manager.search_memories("Agent")
        print(f"   搜索'Agent'找到 {len(search_results)} 条记忆")
        
        # 获取重要记忆
        important_memories = memory_manager.get_important_memories()
        print(f"   重要记忆数量: {len(important_memories)}")
        
        # 获取统计信息
        stats = memory_manager.get_memory_stats()
        print(f"   总记忆数: {stats['total_memories']}")
        print(f"   重要记忆数: {stats['important_memories']}")
        print("   ✅ 记忆系统正常\n")
        
        # 3. 进化引擎演示
        print("🧬 3. 进化引擎演示")
        evolution_engine = EvolutionEngine(memory_manager)
        
        # 模拟一些交互数据
        interaction_data_samples = [
            {
                "response_time": 1.5,
                "user_feedback": "excellent",
                "task_completed": True,
                "error_count": 0
            },
            {
                "response_time": 3.2,
                "user_feedback": "good",
                "task_completed": True,
                "error_count": 1
            },
            {
                "response_time": 0.8,
                "user_feedback": "excellent",
                "task_completed": True,
                "error_count": 0
            },
            {
                "response_time": 5.1,
                "user_feedback": "average",
                "task_completed": False,
                "error_count": 2
            }
        ]
        
        print("   模拟交互数据评估:")
        for i, data in enumerate(interaction_data_samples, 1):
            score = evolution_engine.evaluate_performance(data)
            evolution_engine.update_performance_window(score)
            print(f"   交互 {i}: 评分 {score:.2f}")
        
        # 计算进化指标
        metrics = evolution_engine.calculate_evolution_metrics()
        print(f"   成功率: {metrics.success_rate:.2f}")
        print(f"   响应质量: {metrics.response_quality:.2f}")
        print(f"   学习效率: {metrics.learning_efficiency:.2f}")
        
        # 识别改进领域
        improvement_areas = evolution_engine.identify_improvement_areas(metrics)
        if improvement_areas:
            print(f"   识别的改进领域: {', '.join(improvement_areas)}")
        else:
            print("   当前表现良好，无需改进")
        
        print("   ✅ 进化引擎正常\n")
        
        # 4. 数据持久化演示
        print("💾 4. 数据持久化演示")
        
        # 保存配置
        config.save_config()
        print("   配置已保存到 agent_config.json")
        
        # 导出记忆
        memory_manager.export_memories("demo_memories_export.json")
        print("   记忆已导出到 demo_memories_export.json")
        
        # 保存进化数据
        evolution_engine.save_evolution_data()
        print("   进化数据已保存到 evolution_data.json")
        
        print("   ✅ 数据持久化正常\n")
        
        # 5. 性能统计
        print("📊 5. 系统性能统计")
        evolution_summary = evolution_engine.get_evolution_summary()
        print(f"   进化次数: {evolution_summary.get('total_evolutions', 0)}")
        print(f"   性能趋势: {evolution_summary.get('performance_trend', '数据不足')}")
        
        memory_stats = memory_manager.get_memory_stats()
        print(f"   记忆类型分布: {memory_stats.get('type_distribution', {})}")
        
        print("   ✅ 系统统计正常\n")
        
        print("🎉 核心功能演示完成!")
        print("\n📝 演示总结:")
        print("   - 配置管理: 支持动态更新和持久化")
        print("   - 记忆系统: 可存储、搜索和管理不同类型的记忆")
        print("   - 进化引擎: 能够评估表现并识别改进领域")
        print("   - 数据持久化: 所有数据都能安全保存和加载")
        print("\n💡 要体验完整功能，请:")
        print("   1. 设置 OPENAI_API_KEY 环境变量")
        print("   2. 运行 'python main.py' 开始交互")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理演示数据
        try:
            import os
            if os.path.exists("demo_memory.db"):
                os.remove("demo_memory.db")
                print("\n🧹 演示数据已清理")
        except:
            pass

if __name__ == "__main__":
    simple_demo()