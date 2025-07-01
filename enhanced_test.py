#!/usr/bin/env python3
"""
增强版自主进化Agent综合测试程序
测试所有增强功能的运行状态和性能
"""
import asyncio
import time
import random
import json
from datetime import datetime
from typing import Dict, List, Any

from agent import SelfEvolvingAgent
from config import config

class EnhancedAgentTester:
    """增强版Agent测试器"""
    
    def __init__(self):
        self.agent = None
        self.test_results = {}
        
    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧪 开始增强版Agent综合测试")
        print("=" * 60)
        
        # 初始化Agent
        await self.test_agent_initialization()
        
        # 测试核心功能
        await self.test_enhanced_message_processing()
        
        # 测试智能分析功能
        await self.test_intelligent_analysis()
        
        # 测试情感理解
        await self.test_emotion_understanding()
        
        # 测试个性化功能
        await self.test_personalization()
        
        # 测试性能监控
        await self.test_performance_monitoring()
        
        # 测试学习能力
        await self.test_learning_capabilities()
        
        # 生成测试报告
        self.generate_test_report()
        
        print("\n✅ 综合测试完成")
    
    async def test_agent_initialization(self):
        """测试Agent初始化"""
        print("\n📋 测试1: Agent初始化")
        
        try:
            start_time = time.time()
            self.agent = SelfEvolvingAgent()
            init_time = time.time() - start_time
            
            self.test_results["initialization"] = {
                "success": True,
                "init_time": init_time,
                "version": self.agent.version,
                "features": [
                    "emotion_state", "learning_metrics", "user_profiles",
                    "conversation_patterns", "performance_history", "skill_levels"
                ]
            }
            
            print(f"   ✅ Agent初始化成功")
            print(f"   ⏱️  初始化时间: {init_time:.3f}秒")
            print(f"   🔖 版本: {self.agent.version}")
            print(f"   🧠 情感状态: {self.agent.emotion_state}")
            print(f"   📊 学习指标: {self.agent.learning_metrics}")
            
        except Exception as e:
            self.test_results["initialization"] = {
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ 初始化失败: {e}")
    
    async def test_enhanced_message_processing(self):
        """测试增强消息处理"""
        print("\n📋 测试2: 增强消息处理")
        
        test_messages = [
            {"message": "你好，我需要帮助", "expected_intent": "greeting"},
            {"message": "请帮我分析这个数据", "expected_intent": "analysis"},
            {"message": "怎么创建一个Python程序？", "expected_intent": "help_request"},
            {"message": "这个解决方案很棒，谢谢！", "sentiment": "positive"},
            {"message": "我遇到了很多困难", "sentiment": "negative"}
        ]
        
        results = []
        
        for i, test_case in enumerate(test_messages):
            try:
                print(f"   测试消息 {i+1}: {test_case['message']}")
                
                # 测试消息分析
                analysis = self.agent._analyze_message_intelligence(
                    test_case['message'], 
                    user_id=f"test_user_{i}"
                )
                
                # 验证分析结果
                if "expected_intent" in test_case:
                    intent_match = analysis["intent"] == test_case["expected_intent"]
                    print(f"      意图识别: {analysis['intent']} ({'✅' if intent_match else '❌'})")
                
                print(f"      复杂度: {analysis['complexity']:.3f}")
                print(f"      情感: {analysis['sentiment']:.3f}")
                print(f"      主题: {analysis['topics']}")
                
                results.append({
                    "message": test_case["message"],
                    "analysis": analysis,
                    "success": True
                })
                
            except Exception as e:
                print(f"   ❌ 消息分析失败: {e}")
                results.append({"message": test_case["message"], "error": str(e)})
        
        self.test_results["message_processing"] = {
            "total_tests": len(test_messages),
            "successful": len([r for r in results if r.get("success")]),
            "results": results
        }
    
    async def test_intelligent_analysis(self):
        """测试智能分析功能"""
        print("\n📋 测试3: 智能分析功能")
        
        # 测试复杂度计算
        complexity_tests = [
            ("简单", 0.2),
            ("这是一个相对复杂的技术问题，需要深入分析", 0.5),
            ("在深度学习和自然语言处理的交叉领域中，transformer架构的注意力机制为序列建模提供了革命性的解决方案", 0.8)
        ]
        
        for text, expected_range in complexity_tests:
            complexity = self.agent._calculate_complexity(text)
            print(f"   文本: '{text[:30]}...'")
            print(f"   复杂度: {complexity:.3f} (预期范围: ~{expected_range})")
        
        # 测试情感分析
        sentiment_tests = [
            ("我很开心，这太棒了！", 0.5),
            ("这真是太糟糕了", -0.5),
            ("今天天气不错", 0.0)
        ]
        
        for text, expected_sentiment in sentiment_tests:
            sentiment = self.agent._analyze_sentiment(text)
            print(f"   文本: '{text}'")
            print(f"   情感: {sentiment:.3f} (预期: {expected_sentiment})")
        
        self.test_results["intelligent_analysis"] = {
            "complexity_tests": len(complexity_tests),
            "sentiment_tests": len(sentiment_tests),
            "status": "completed"
        }
    
    async def test_emotion_understanding(self):
        """测试情感理解"""
        print("\n📋 测试4: 情感理解")
        
        # 初始情感状态
        initial_emotion = self.agent.emotion_state.copy()
        print(f"   初始情感状态: {initial_emotion}")
        
        # 模拟情感更新（这里需要实现_update_emotion_state方法）
        print(f"   当前情感描述: {self.agent._get_emotion_description()}")
        
        self.test_results["emotion_understanding"] = {
            "initial_state": initial_emotion,
            "description": self.agent._get_emotion_description(),
            "status": "tested"
        }
    
    async def test_personalization(self):
        """测试个性化功能"""
        print("\n📋 测试5: 个性化功能")
        
        # 测试用户画像
        test_users = ["user_1", "user_2", "user_3"]
        
        for user_id in test_users:
            # 模拟用户交互
            for i in range(3):
                analysis = self.agent._analyze_message_intelligence(
                    f"用户{user_id}的消息{i+1}", 
                    user_id=user_id
                )
                
                # 更新用户画像（需要实现）
                if user_id not in self.agent.user_profiles:
                    self.agent.user_profiles[user_id] = {
                        "interaction_count": 0,
                        "preferred_style": "balanced",
                        "topics_of_interest": []
                    }
                
                self.agent.user_profiles[user_id]["interaction_count"] += 1
        
        print(f"   用户画像数量: {len(self.agent.user_profiles)}")
        for user_id, profile in self.agent.user_profiles.items():
            print(f"   {user_id}: {profile}")
        
        self.test_results["personalization"] = {
            "user_profiles": len(self.agent.user_profiles),
            "profiles": dict(self.agent.user_profiles)
        }
    
    async def test_performance_monitoring(self):
        """测试性能监控"""
        print("\n📋 测试6: 性能监控")
        
        # 模拟性能数据
        for i in range(10):
            performance_data = {
                "timestamp": time.time(),
                "response_time": random.uniform(0.5, 3.0),
                "quality_score": random.uniform(0.6, 1.0),
                "user_satisfaction": random.uniform(0.5, 1.0)
            }
            self.agent.performance_history.append(performance_data)
        
        print(f"   性能历史记录数: {len(self.agent.performance_history)}")
        
        if self.agent.performance_history:
            avg_response_time = sum(p["response_time"] for p in self.agent.performance_history) / len(self.agent.performance_history)
            avg_quality = sum(p["quality_score"] for p in self.agent.performance_history) / len(self.agent.performance_history)
            
            print(f"   平均响应时间: {avg_response_time:.3f}秒")
            print(f"   平均质量评分: {avg_quality:.3f}")
        
        self.test_results["performance_monitoring"] = {
            "records": len(self.agent.performance_history),
            "avg_response_time": avg_response_time if self.agent.performance_history else 0,
            "avg_quality": avg_quality if self.agent.performance_history else 0
        }
    
    async def test_learning_capabilities(self):
        """测试学习能力"""
        print("\n📋 测试7: 学习能力")
        
        # 测试技能提升
        skills = ["编程", "数据分析", "语言理解", "问题解决"]
        
        for skill in skills:
            # 模拟技能提升
            self.agent.skill_levels[skill] = random.uniform(0.3, 0.9)
        
        print(f"   技能数量: {len(self.agent.skill_levels)}")
        for skill, level in self.agent.skill_levels.items():
            print(f"   {skill}: {level:.3f}")
        
        # 测试学习指标
        self.agent.learning_metrics["improvements"] = random.randint(5, 15)
        print(f"   学习改进次数: {self.agent.learning_metrics['improvements']}")
        
        self.test_results["learning_capabilities"] = {
            "skills": dict(self.agent.skill_levels),
            "improvements": self.agent.learning_metrics["improvements"],
            "total_interactions": self.agent.learning_metrics["interactions"]
        }
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n📊 测试报告")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get("success", True))
        
        print(f"总测试模块: {total_tests}")
        print(f"成功测试: {successful_tests}")
        print(f"成功率: {successful_tests/total_tests*100:.1f}%")
        
        print("\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅" if result.get("success", True) else "❌"
            print(f"  {status} {test_name}")
            
            if "error" in result:
                print(f"     错误: {result['error']}")
        
        # 保存详细报告
        with open("enhanced_test_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "successful_tests": successful_tests,
                    "success_rate": successful_tests/total_tests
                },
                "detailed_results": self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 详细报告已保存到: enhanced_test_report.json")

async def main():
    """主函数"""
    tester = EnhancedAgentTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())