#!/usr/bin/env python3
"""
通信协议测试模块 - 自主进化Agent系统
测试Agent间通信功能
"""

import unittest
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_evolutionary_agent_system import (
    CommunicationProtocol,
    ResearcherAgent, ExecutorAgent
)


class TestCommunicationProtocol(unittest.TestCase):
    """测试通信协议基础功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
    
    def test_protocol_initialization(self):
        """测试协议初始化"""
        self.assertIsInstance(self.communication, CommunicationProtocol)
        self.assertEqual(len(self.communication.messages), 0)
        self.assertEqual(len(self.communication.subscribers), 0)
    
    def test_message_publishing(self):
        """测试消息发布"""
        topic = "test_topic"
        message = {"content": "test message", "priority": "high"}
        sender = "agent_001"
        
        # 发布消息
        self.communication.publish(topic, message, sender)
        
        # 检查消息是否被正确存储
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['message'], message)
        self.assertEqual(messages[0]['sender'], sender)
        self.assertIn('timestamp', messages[0])
    
    def test_multiple_messages(self):
        """测试多条消息发布"""
        topic = "multi_test"
        
        # 发布多条消息
        for i in range(5):
            message = {"id": i, "content": f"message_{i}"}
            self.communication.publish(topic, message, f"agent_{i}")
        
        # 检查消息数量和顺序
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 5)
        
        for i, msg in enumerate(messages):
            self.assertEqual(msg['message']['id'], i)
            self.assertEqual(msg['sender'], f"agent_{i}")
    
    def test_topic_isolation(self):
        """测试主题隔离"""
        # 向不同主题发布消息
        self.communication.publish("topic_a", {"data": "a"}, "agent1")
        self.communication.publish("topic_b", {"data": "b"}, "agent2")
        
        # 检查主题隔离
        messages_a = self.communication.get_messages("topic_a")
        messages_b = self.communication.get_messages("topic_b")
        
        self.assertEqual(len(messages_a), 1)
        self.assertEqual(len(messages_b), 1)
        self.assertEqual(messages_a[0]['message']['data'], "a")
        self.assertEqual(messages_b[0]['message']['data'], "b")
    
    def test_subscription(self):
        """测试消息订阅"""
        topic = "subscription_test"
        subscriber = "agent_subscriber"
        
        # 订阅主题
        self.communication.subscribe(topic, subscriber)
        
        # 检查订阅者列表
        subscribers = self.communication.subscribers.get(topic, [])
        self.assertIn(subscriber, subscribers)
    
    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        topic = "multi_subscriber_test"
        subscribers = ["agent1", "agent2", "agent3"]
        
        # 多个Agent订阅同一主题
        for subscriber in subscribers:
            self.communication.subscribe(topic, subscriber)
        
        # 检查所有订阅者
        topic_subscribers = self.communication.subscribers.get(topic, [])
        for subscriber in subscribers:
            self.assertIn(subscriber, topic_subscribers)
    
    def test_get_nonexistent_topic(self):
        """测试获取不存在的主题"""
        messages = self.communication.get_messages("nonexistent_topic")
        self.assertEqual(len(messages), 0)


class TestAgentCommunication(unittest.TestCase):
    """测试Agent间通信"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
        self.researcher = ResearcherAgent("researcher_001", self.communication)
        self.executor = ExecutorAgent("executor_001", self.communication)
    
    def test_agent_message_sending(self):
        """测试Agent发送消息"""
        topic = "task_assignment"
        message = {
            "task_type": "data_analysis",
            "deadline": "2024-01-01",
            "priority": "high"
        }
        
        # 研究者发送消息
        self.communication.publish(topic, message, self.researcher.agent_id)
        
        # 检查消息
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['sender'], self.researcher.agent_id)
        self.assertEqual(messages[0]['message']['task_type'], "data_analysis")
    
    def test_agent_subscription_workflow(self):
        """测试Agent订阅工作流"""
        topic = "task_updates"
        
        # 执行者订阅任务更新
        self.communication.subscribe(topic, self.executor.agent_id)
        
        # 研究者发布任务更新
        update_message = {
            "status": "in_progress",
            "completion": 0.5,
            "notes": "数据收集完成"
        }
        self.communication.publish(topic, update_message, self.researcher.agent_id)
        
        # 检查执行者是否在订阅者列表中
        subscribers = self.communication.subscribers.get(topic, [])
        self.assertIn(self.executor.agent_id, subscribers)
        
        # 检查消息是否正确发布
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['message']['status'], "in_progress")
    
    def test_bidirectional_communication(self):
        """测试双向通信"""
        request_topic = "task_requests"
        response_topic = "task_responses"
        
        # 研究者订阅响应
        self.communication.subscribe(response_topic, self.researcher.agent_id)
        
        # 执行者订阅请求
        self.communication.subscribe(request_topic, self.executor.agent_id)
        
        # 研究者发送任务请求
        request = {
            "request_id": "req_001",
            "task": "process_dataset",
            "parameters": {"format": "csv"}
        }
        self.communication.publish(request_topic, request, self.researcher.agent_id)
        
        # 执行者发送响应
        response = {
            "request_id": "req_001",
            "status": "accepted",
            "estimated_time": "30 minutes"
        }
        self.communication.publish(response_topic, response, self.executor.agent_id)
        
        # 检查双向通信
        requests = self.communication.get_messages(request_topic)
        responses = self.communication.get_messages(response_topic)
        
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(requests[0]['message']['request_id'], "req_001")
        self.assertEqual(responses[0]['message']['request_id'], "req_001")


class TestCommunicationPatterns(unittest.TestCase):
    """测试通信模式"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
    
    def test_broadcast_pattern(self):
        """测试广播模式"""
        topic = "system_announcement"
        
        # 多个Agent订阅系统公告
        agents = ["agent1", "agent2", "agent3", "agent4"]
        for agent in agents:
            self.communication.subscribe(topic, agent)
        
        # 系统管理员发布公告
        announcement = {
            "type": "system_update",
            "message": "系统将在30分钟后重启",
            "severity": "warning"
        }
        self.communication.publish(topic, announcement, "system_admin")
        
        # 检查所有Agent都能收到公告
        subscribers = self.communication.subscribers.get(topic, [])
        for agent in agents:
            self.assertIn(agent, subscribers)
        
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['message']['type'], "system_update")
    
    def test_request_response_pattern(self):
        """测试请求-响应模式"""
        request_topic = "service_requests"
        response_topic = "service_responses"
        
        # 服务提供者订阅请求
        self.communication.subscribe(request_topic, "service_provider")
        
        # 客户端订阅响应
        self.communication.subscribe(response_topic, "client")
        
        # 客户端发送服务请求
        service_request = {
            "service_id": "data_processing",
            "client_id": "client",
            "parameters": {"input_file": "data.csv"}
        }
        self.communication.publish(request_topic, service_request, "client")
        
        # 服务提供者发送响应
        service_response = {
            "service_id": "data_processing",
            "client_id": "client",
            "result": "success",
            "output_file": "processed_data.csv"
        }
        self.communication.publish(response_topic, service_response, "service_provider")
        
        # 验证请求-响应链
        requests = self.communication.get_messages(request_topic)
        responses = self.communication.get_messages(response_topic)
        
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(requests[0]['message']['client_id'], "client")
        self.assertEqual(responses[0]['message']['client_id'], "client")
    
    def test_event_driven_pattern(self):
        """测试事件驱动模式"""
        event_topic = "system_events"
        
        # 多个监听器订阅事件
        listeners = ["logger", "monitor", "alert_system"]
        for listener in listeners:
            self.communication.subscribe(event_topic, listener)
        
        # 发布系统事件
        events = [
            {"event": "agent_started", "agent_id": "agent_001"},
            {"event": "task_completed", "task_id": "task_123"},
            {"event": "error_occurred", "error_code": "E001"}
        ]
        
        for event in events:
            self.communication.publish(event_topic, event, "event_publisher")
        
        # 检查事件发布
        published_events = self.communication.get_messages(event_topic)
        self.assertEqual(len(published_events), 3)
        
        # 检查所有监听器都订阅了事件
        subscribers = self.communication.subscribers.get(event_topic, [])
        for listener in listeners:
            self.assertIn(listener, subscribers)


class TestCommunicationReliability(unittest.TestCase):
    """测试通信可靠性"""
    
    def setUp(self):
        """设置测试环境"""
        self.communication = CommunicationProtocol()
    
    def test_message_ordering(self):
        """测试消息顺序"""
        topic = "ordered_messages"
        
        # 按顺序发送消息
        for i in range(10):
            message = {"sequence": i, "data": f"message_{i}"}
            self.communication.publish(topic, message, "sender")
        
        # 检查消息顺序
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 10)
        
        for i, msg in enumerate(messages):
            self.assertEqual(msg['message']['sequence'], i)
    
    def test_large_message_handling(self):
        """测试大消息处理"""
        topic = "large_messages"
        
        # 发送大消息
        large_data = "x" * 10000  # 10KB的数据
        large_message = {
            "type": "large_data",
            "data": large_data,
            "size": len(large_data)
        }
        
        self.communication.publish(topic, large_message, "sender")
        
        # 检查大消息是否正确处理
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]['message']['data']), 10000)
    
    def test_concurrent_publishing(self):
        """测试并发发布"""
        topic = "concurrent_test"
        
        # 模拟多个发送者同时发布消息
        senders = [f"sender_{i}" for i in range(5)]
        
        for sender in senders:
            for j in range(3):
                message = {"sender": sender, "message_num": j}
                self.communication.publish(topic, message, sender)
        
        # 检查所有消息都被正确接收
        messages = self.communication.get_messages(topic)
        self.assertEqual(len(messages), 15)  # 5个发送者 × 3条消息
        
        # 检查每个发送者的消息
        sender_counts = {}
        for msg in messages:
            sender = msg['sender']
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        for sender in senders:
            self.assertEqual(sender_counts[sender], 3)


if __name__ == '__main__':
    print("🧪 运行通信协议测试套件")
    print("=" * 50)
    
    # 运行测试
    unittest.main(verbosity=2)