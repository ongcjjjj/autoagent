"""
自我进化Agent演示脚本
"""
import asyncio
import os
from agent import SelfEvolvingAgent

async def demo():
    """演示脚本"""
    print("🚀 自我进化Agent演示")
    print("=" * 50)
    
    # 检查环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 请先设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY=your_api_key_here")
        return
    
    try:
        # 创建Agent实例
        print("📊 正在初始化Agent...")
        agent = SelfEvolvingAgent(name="DemoAgent")
        
        # 测试连接
        print("🔗 测试API连接...")
        if not await agent.test_connection():
            print("❌ API连接失败，请检查配置")
            return
        
        print("✅ 连接成功！\n")
        
        # 演示对话
        demo_messages = [
            "你好，请介绍一下自己",
            "你有什么特殊能力？",
            "如何配置你的参数？",
            "你是如何学习和进化的？"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"🧑 用户 {i}: {message}")
            
            # 处理消息
            response = await agent.process_message(message)
            
            if response.get("error"):
                print(f"❌ 错误: {response['content']}")
            else:
                print(f"🤖 {agent.name}: {response['content']}")
                print(f"   (响应时间: {response.get('request_time', 0):.2f}s)")
            
            print("-" * 50)
        
        # 显示状态信息
        print("\n📈 Agent状态:")
        status = agent.get_status()
        
        print(f"   名称: {status['agent']['name']}")
        print(f"   版本: {status['agent']['version']}")
        print(f"   总记忆数: {status['memory']['total_memories']}")
        print(f"   重要记忆数: {status['memory']['important_memories']}")
        
        evolution_info = status['evolution']
        if "message" not in evolution_info:
            print(f"   进化次数: {evolution_info['total_evolutions']}")
            print(f"   性能趋势: {evolution_info['performance_trend']}")
        else:
            print(f"   进化状态: {evolution_info['message']}")
        
        # 搜索记忆演示
        print("\n🔍 记忆搜索演示:")
        memories = agent.search_memory("介绍")
        print(f"   搜索'介绍'找到 {len(memories)} 条记忆")
        
        if memories:
            latest_memory = memories[0]
            print(f"   最新记忆: {latest_memory['content'][:50]}...")
        
        # 手动添加记忆
        print("\n💾 添加手动记忆:")
        memory_id = agent.add_manual_memory(
            "这是一次成功的演示运行",
            memory_type="demo",
            importance=0.8,
            tags=["demo", "success"]
        )
        print(f"   记忆已添加，ID: {memory_id}")
        
        # 导出数据演示
        print("\n📤 导出数据:")
        agent.export_data("demo_export.json")
        print("   数据已导出到 demo_export.json")
        
        print("\n🎉 演示完成！")
        print("运行 'python main.py' 开始交互式体验")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo())