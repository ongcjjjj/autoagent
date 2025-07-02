#!/usr/bin/env python3
"""
简化测试模块 - 自主进化Agent系统
测试系统的基本功能
"""

import unittest
import asyncio
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    AutonomousEvolutionarySystem,
    AgentRole, CommunicationProtocol,
    ResearcherAgent, ExecutorAgent, CriticAgent,
    CoordinatorAgent, ArchitectAgent,
    AgentAction, ActionType
)


class TestSystemBasics(unittest.TestCase):
    """测试系统基础功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = AutonomousEvolutionarySystem()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsInstance(self.system, AutonomousEvolutionarySystem)
        self.assertIsInstance(self.system.communication, CommunicationProtocol)
        self.assertEqual(len(self.system.agents), 0)
    
    def test_create_standard_team(self):
        """测试创建标准团队"""
        team = self.system.create_standard_team()
        
        # 检查团队组成
        self.assertGreater(len(team), 0)
        self.assertIn('researcher', team)
        self.assertIn('executor', team)
        self.assertIn('critic', team)
        self.assertIn('coordinator', team)
        self.assertIn('architect', team)
        
        # 检查Agent类型
        self.assertIsInstance(team['researcher'], ResearcherAgent)
        self.assertIsInstance(team['executor'], ExecutorAgent)
        self.assertIsInstance(team['critic'], CriticAgent)
        self.assertIsInstance(team['coordinator'], CoordinatorAgent)
        self.assertIsInstance(team['architect'], ArchitectAgent)
    
    def test_add_agent(self):
        """测试添加Agent"""
        communication = CommunicationProtocol()
        agent = ResearcherAgent("test_researcher", communication)
        
        self.system.add_agent(agent)
        
        self.assertEqual(len(self.system.agents), 1)
        self.assertIn("test_researcher", self.system.agents)


class TestAgentCreation(unittest.TestCase):
    """测试Agent创建"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
    
    def test_researcher_creation(self):
        """测试研究者Agent创建"""
        agent = ResearcherAgent("researcher_01", self.communication)
        
        self.assertEqual(agent.agent_id, "researcher_01")
        self.assertEqual(agent.role, AgentRole.RESEARCHER)
        self.assertIsInstance(agent.communication, CommunicationProtocol)
        self.assertEqual(len(agent.action_history), 0)
        self.assertEqual(len(agent.memory), 0)
    
    def test_executor_creation(self):
        """测试执行者Agent创建"""
        agent = ExecutorAgent("executor_01", self.communication)
        
        self.assertEqual(agent.agent_id, "executor_01")
        self.assertEqual(agent.role, AgentRole.EXECUTOR)
        self.assertIsInstance(agent.communication, CommunicationProtocol)
    
    def test_critic_creation(self):
        """测试评判者Agent创建"""
        agent = CriticAgent("critic_01", self.communication)
        
        self.assertEqual(agent.agent_id, "critic_01")
        self.assertEqual(agent.role, AgentRole.CRITIC)
        self.assertIsInstance(agent.communication, CommunicationProtocol)
    
    def test_coordinator_creation(self):
        """测试协调者Agent创建"""
        agent = CoordinatorAgent("coordinator_01", self.communication)
        
        self.assertEqual(agent.agent_id, "coordinator_01")
        self.assertEqual(agent.role, AgentRole.COORDINATOR)
        self.assertIsInstance(agent.communication, CommunicationProtocol)
    
    def test_architect_creation(self):
        """测试架构师Agent创建"""
        agent = ArchitectAgent("architect_01", self.communication)
        
        self.assertEqual(agent.agent_id, "architect_01")
        self.assertEqual(agent.role, AgentRole.OPTIMIZER)
        self.assertIsInstance(agent.communication, CommunicationProtocol)


class TestCommunicationProtocol(unittest.TestCase):
    """测试通信协议"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
    
    def test_message_publishing(self):
        """测试消息发布"""
        message = {
            'type': 'task_request',
            'content': 'analyze dataset',
            'priority': 'high'
        }
        
        # 发布消息
        self.communication.publish('task_requests', message, 'agent_01')
        
        # 检查消息是否被记录
        messages = self.communication.get_messages('task_requests')
        self.assertGreater(len(messages), 0)
        self.assertEqual(messages[-1]['message'], message)
        self.assertEqual(messages[-1]['sender'], 'agent_01')
    
    def test_message_subscription(self):
        """测试消息订阅"""
        # 订阅消息
        self.communication.subscribe('task_requests', 'agent_02')
        
        # 检查订阅者列表
        subscribers = self.communication.subscribers.get('task_requests', [])
        self.assertIn('agent_02', subscribers)
    
    def test_get_messages(self):
        """测试获取消息"""
        # 发布几条消息
        for i in range(3):
            message = {'id': i, 'content': f'message_{i}'}
            self.communication.publish('test_topic', message, f'agent_{i}')
        
        # 获取消息
        messages = self.communication.get_messages('test_topic')
        self.assertEqual(len(messages), 3)
        
        # 检查消息内容
        for i, msg in enumerate(messages):
            self.assertEqual(msg['message']['id'], i)
            self.assertEqual(msg['sender'], f'agent_{i}')


class TestAsyncFunctionality(unittest.TestCase):
    """测试异步功能"""
    
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
        self.assertIsNotNone(metrics)
        self.assertHasAttribute(metrics, 'composite_score')
        self.assertGreaterEqual(metrics.composite_score, 0)
        self.assertLessEqual(metrics.composite_score, 1)
    
    async def test_collaborative_task(self):
        """测试协作任务"""
        # 创建团队
        team = self.system.create_standard_team()
        
        # 运行简单任务
        result = await self.system.run_collaborative_task(
            goal="测试任务",
            max_cycles=1
        )
        
        # 检查结果
        self.assertIsInstance(result, dict)
        self.assertIn('total_cycles', result)
        self.assertIn('total_actions', result)
        self.assertGreaterEqual(result['total_cycles'], 0)
        self.assertGreaterEqual(result['total_actions'], 0)
    
    def assertHasAttribute(self, obj, attr_name):
        """检查对象是否有指定属性"""
        self.assertTrue(hasattr(obj, attr_name), 
                       f"Object {obj} does not have attribute '{attr_name}'")


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
TestAsyncFunctionality.test_system_performance_evaluation = run_async_test(
    TestAsyncFunctionality.test_system_performance_evaluation
)
TestAsyncFunctionality.test_collaborative_task = run_async_test(
    TestAsyncFunctionality.test_collaborative_task
)


if __name__ == '__main__':
    print("🧪 运行简化测试套件")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)