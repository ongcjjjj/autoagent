#!/usr/bin/env python3
"""
基础使用示例 - 自主进化Agent系统
展示系统的基本功能和使用方法
"""

import asyncio
import sys
import os

# 添加父目录到路径，以便导入主模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    AutonomousEvolutionarySystem,
    AgentRole
)


async def basic_example():
    """基础使用示例"""
    print("🚀 基础使用示例 - 自主进化Agent系统")
    print("=" * 50)
    
    # 1. 创建系统
    print("\n📋 步骤1: 创建自主进化系统")
    system = AutonomousEvolutionarySystem()
    print("✅ 系统创建成功")
    
    # 2. 创建标准团队
    print("\n👥 步骤2: 创建标准Agent团队")
    team = system.create_standard_team()
    print(f"✅ 创建了 {len(team)} 个专业Agent:")
    for role, agent in team.items():
        print(f"   - {role}: {agent.agent_id} ({agent.role.value})")
    
    # 3. 运行简单任务
    print("\n🎯 步骤3: 运行协作任务")
    task_goal = "分析并优化一个简单的数据处理流程"
    print(f"任务目标: {task_goal}")
    
    result = await system.run_collaborative_task(
        goal=task_goal,
        max_cycles=3  # 简化演示，只运行3个周期
    )
    
    # 4. 显示结果
    print("\n📊 步骤4: 任务执行结果")
    print(f"   - 执行周期数: {result['total_cycles']}")
    print(f"   - 总行动数: {result['total_actions']}")
    print(f"   - 系统进化次数: {result['evolution_cycles']}")
    
    if result['final_metrics']:
        metrics = result['final_metrics']
        print(f"\n📈 最终性能指标:")
        print(f"   - 综合得分: {metrics.composite_score:.3f}")
        print(f"   - 可训练性: {metrics.trainability:.3f}")
        print(f"   - 泛化能力: {metrics.generalization:.3f}")
        print(f"   - 创造性: {metrics.creativity_score:.3f}")
        print(f"   - 协作效率: {metrics.collaboration_efficiency:.3f}")
    
    # 5. 保存系统状态
    print("\n💾 步骤5: 保存系统状态")
    state_file = "data/system_states/basic_example_state.pkl"
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    system.save_system_state(state_file)
    print(f"✅ 系统状态已保存到: {state_file}")
    
    print("\n🎉 基础示例执行完成！")


async def monitoring_example():
    """系统监控示例"""
    print("\n" + "=" * 50)
    print("📊 系统监控示例")
    print("=" * 50)
    
    # 创建系统并运行任务
    system = AutonomousEvolutionarySystem()
    team = system.create_standard_team()
    
    # 运行多个小任务来展示监控功能
    tasks = [
        "数据预处理优化",
        "算法性能调优", 
        "系统架构分析"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n🎯 任务 {i}: {task}")
        
        result = await system.run_collaborative_task(
            goal=task,
            max_cycles=2
        )
        
        # 显示系统状态
        system_metrics = await system.evaluate_system_performance()
        print(f"   系统性能: {system_metrics.composite_score:.3f}")
        
        # 显示Agent状态
        print(f"   Agent状态:")
        for role, agent in team.items():
            print(f"     - {role}: 优化{agent.optimization_counter}次, "
                  f"温度{agent.temperature:.2f}, "
                  f"记忆{len(agent.memory)}条")
    
    print(f"\n📈 系统进化历史 ({len(system.system_metrics)} 个评估点):")
    for i, record in enumerate(system.system_metrics, 1):
        metrics = record['metrics']
        timestamp = record['timestamp'].strftime("%H:%M:%S")
        print(f"   {i}. {timestamp}: 得分 {metrics.composite_score:.3f}")


if __name__ == "__main__":
    print("🤖 自主进化Agent系统 - 基础使用示例")
    print("这个示例展示了系统的核心功能和基本使用方法\n")
    
    # 创建数据目录
    os.makedirs("data/system_states", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    
    # 运行示例
    asyncio.run(basic_example())
    asyncio.run(monitoring_example())
    
    print("\n" + "=" * 50)
    print("💡 提示:")
    print("- 查看 'data/system_states/' 目录中的保存状态")
    print("- 运行 'python examples/custom_agent.py' 查看自定义Agent示例")
    print("- 运行 'python examples/advanced_config.py' 查看高级配置示例")
    print("=" * 50)