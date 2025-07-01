"""
最小化测试脚本
只使用Python标准库，测试核心功能
"""
import sys
import os
import time

# 添加当前目录到Python路径
sys.path.insert(0, '.')

def test_basic_functionality():
    """测试基础功能"""
    print("🚀 自我进化Agent最小化测试")
    print("=" * 50)
    
    try:
        # 1. 测试最小配置模块
        print("📋 1. 测试配置模块")
        from config_minimal import Config, OpenAIConfig, AgentConfig
        
        config = Config()
        print(f"   默认模型: {config.openai_config.model}")
        print(f"   Agent名称: {config.agent_config.name}")
        
        # 更新配置
        config.update_openai_config(model="gpt-4", temperature=0.8)
        print(f"   更新后模型: {config.openai_config.model}")
        print("   ✅ 配置模块正常\n")
        
        # 2. 测试记忆模块
        print("🧠 2. 测试记忆模块")
        from memory import MemoryManager, Memory
        
        # 创建临时数据库
        memory_manager = MemoryManager("test_memory.db")
        
        # 添加测试记忆
        test_memory = Memory(
            content="这是一个测试记忆",
            memory_type="test",
            importance=0.8,
            tags=["test", "demo"]
        )
        
        memory_id = memory_manager.add_memory(test_memory)
        print(f"   添加记忆 ID: {memory_id}")
        
        # 搜索记忆
        results = memory_manager.search_memories("测试")
        print(f"   搜索结果数量: {len(results)}")
        
        # 获取统计信息
        stats = memory_manager.get_memory_stats()
        print(f"   总记忆数: {stats['total_memories']}")
        print("   ✅ 记忆模块正常\n")
        
        # 3. 测试进化模块
        print("🧬 3. 测试进化模块")
        from evolution import EvolutionEngine
        
        evolution_engine = EvolutionEngine(memory_manager)
        
        # 添加测试数据
        test_data = {
            "response_time": 1.5,
            "user_feedback": "excellent",
            "task_completed": True,
            "error_count": 0
        }
        
        score = evolution_engine.evaluate_performance(test_data)
        print(f"   性能评分: {score:.2f}")
        
        evolution_engine.update_performance_window(score)
        
        metrics = evolution_engine.calculate_evolution_metrics()
        print(f"   成功率: {metrics.success_rate:.2f}")
        print("   ✅ 进化模块正常\n")
        
        # 4. 测试数据持久化
        print("💾 4. 测试数据持久化")
        
        # 保存配置
        config.save_config()
        print("   配置已保存")
        
        # 导出记忆
        memory_manager.export_memories("test_export.json")
        print("   记忆已导出")
        
        # 保存进化数据
        evolution_engine.save_evolution_data()
        print("   进化数据已保存")
        print("   ✅ 数据持久化正常\n")
        
        print("🎉 所有测试通过!")
        print("\n📝 文件结构:")
        
        files = [
            "config_minimal.py", "memory.py", "evolution.py", 
            "openai_client.py", "agent.py", "main.py"
        ]
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✓ {file} ({size} bytes)")
            else:
                print(f"   ✗ {file} (不存在)")
        
        print("\n💡 下一步:")
        print("1. 安装依赖: pip install openai python-dotenv")
        print("2. 设置环境变量: export OPENAI_API_KEY=your_key")
        print("3. 运行完整版本: python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        cleanup_files = [
            "test_memory.db", "test_export.json", 
            "agent_config.json", "evolution_data.json"
        ]
        
        for file in cleanup_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
        
        print("\n🧹 测试文件已清理")

if __name__ == "__main__":
    test_basic_functionality()