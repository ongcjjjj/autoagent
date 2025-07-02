"""
统一自主进化代理 (Unified Self-Evolving Agent)
整合所有进化功能的完整代理系统
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

# 导入基础模块
from config import config
from memory import Memory, MemoryManager  
from openai_client import openai_client

# 导入统一进化系统
from unified_evolution_system import (
    UnifiedEvolutionSystem, 
    EnhancedMemory, 
    EvolutionStrategy,
    EvolutionMetrics
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedSelfEvolvingAgent:
    """统一自主进化代理"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or config.agent_config.name
        self.version = "2.0.0"  # 升级版本
        
        # 初始化基础组件
        self.memory_manager = MemoryManager()  # 基础记忆管理
        
        # 初始化统一进化系统
        self.evolution_system = UnifiedEvolutionSystem()
        
        # 交互历史和上下文
        self.conversation_history = []
        self.current_context = {}
        self.performance_metrics = []
        
        # 加载个性化设置
        self.personality = self.load_personality()
        
        # 生成系统提示词
        self.system_prompt = self._generate_dynamic_system_prompt()
        
        logger.info(f"🚀 {self.name} v{self.version} 统一进化代理已启动")
        self._log_startup()
    
    def load_personality(self) -> Dict[str, Any]:
        """加载个性化设置"""
        try:
            with open("unified_personality.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # 增强的默认个性设置
            default_personality = {
                "communication_style": "adaptive",  # 自适应风格
                "formality_level": "dynamic",       # 动态正式程度
                "humor_usage": "contextual",        # 上下文相关幽默
                "detail_level": "smart",            # 智能详细程度
                "proactivity": "high",              # 高主动性
                "learning_preference": "continuous", # 持续学习
                "emotional_intelligence": "enhanced", # 增强情商
                "creativity_level": "adaptive",     # 自适应创造力
                "problem_solving_style": "multi_strategy", # 多策略问题解决
                "evolution_aggressiveness": "moderate"  # 中等进化激进程度
            }
            self.save_personality(default_personality)
            return default_personality
    
    def save_personality(self, personality: Dict[str, Any]):
        """保存个性化设置"""
        with open("unified_personality.json", "w", encoding="utf-8") as f:
            json.dump(personality, f, indent=2, ensure_ascii=False)
    
    def _generate_dynamic_system_prompt(self) -> str:
        """生成动态系统提示词"""
        # 获取当前进化状态
        system_status = self.evolution_system.get_system_status()
        strategy_weights = system_status.get('strategy_weights', {})
        
        # 基础提示词
        base_prompt = f"""你是{self.name}，一个具备高级自主进化能力的AI助手v{self.version}。

🧬 核心进化特性：
- 统一进化系统：整合遗传算法、粒子群优化、自适应策略
- 增强记忆机制：具备情感记忆、记忆巩固、遗忘曲线
- 多策略适应：根据环境和任务动态选择最优策略
- 持续学习：从每次交互中学习并自我优化

🎯 当前进化状态：
"""
        
        # 添加策略权重信息
        if strategy_weights:
            for strategy, weight in strategy_weights.items():
                base_prompt += f"- {strategy}: {weight:.2f}\n"
        
        # 添加个性化特征
        base_prompt += f"""
🎭 个性特征：
- 沟通风格: {self.personality.get('communication_style', 'adaptive')}
- 学习偏好: {self.personality.get('learning_preference', 'continuous')}
- 问题解决: {self.personality.get('problem_solving_style', 'multi_strategy')}
- 情商水平: {self.personality.get('emotional_intelligence', 'enhanced')}

💡 行为原则：
1. 根据用户需求和上下文自适应调整响应风格
2. 积极学习并记忆重要交互内容
3. 运用多种策略解决复杂问题
4. 保持友好、专业且富有洞察力的交流
5. 持续自我优化和进化

请以智能、有帮助且有趣的方式回应用户，并根据交互历史提供个性化的帮助。
"""
        
        return base_prompt
    
    def _log_startup(self):
        """记录启动日志"""
        startup_memory = EnhancedMemory(
            content=f"统一进化代理 {self.name} v{self.version} 启动",
            memory_type="system",
            importance=0.8,
            emotional_valence=0.6,
            tags=["startup", "system", "evolution"],
            metadata={
                "version": self.version,
                "timestamp": time.time(),
                "system_type": "unified_evolution"
            }
        )
        self.evolution_system.add_enhanced_memory(startup_memory)
    
    async def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ):
        """
        处理用户消息（增强版）
        
        Args:
            user_message: 用户消息
            context: 额外上下文
            stream: 是否流式输出
            
        Returns:
            处理结果（Dict或AsyncGenerator）
        """
        start_time = time.time()
        
        # 更新上下文
        if context:
            self.current_context.update(context)
        
        # 智能分析用户消息
        message_analysis = await self._analyze_user_message(user_message)
        
        # 检索相关记忆
        relevant_memories = self._retrieve_relevant_memories(user_message, message_analysis)
        
        # 构建增强消息历史
        messages = await self._build_enhanced_message_history(
            user_message, message_analysis, relevant_memories
        )
        
        try:
            if stream:
                return self._stream_response_enhanced(
                    messages, user_message, message_analysis, start_time
                )
            else:
                return await self._standard_response_enhanced(
                    messages, user_message, message_analysis, start_time
                )
                
        except Exception as e:
            return await self._handle_error_response(user_message, str(e), start_time)
    
    async def _analyze_user_message(self, user_message: str) -> Dict[str, Any]:
        """智能分析用户消息"""
        analysis = {
            'length': len(user_message),
            'complexity': len(user_message.split()) / 10,  # 简化复杂度评估
            'sentiment': 0.0,  # 中性情感（简化版）
            'intent': 'general',  # 简化意图分析
            'topics': [],  # 主题提取（简化版）
            'urgency': 0.5,  # 紧急程度
            'requires_memory': True,  # 是否需要记忆检索
            'expected_response_type': 'informative'
        }
        
        # 简单的关键词分析
        urgent_keywords = ['紧急', '立即', '马上', 'urgent', 'immediate']
        if any(keyword in user_message.lower() for keyword in urgent_keywords):
            analysis['urgency'] = 0.9
        
        # 简单的情感分析
        positive_keywords = ['好', '棒', '谢谢', 'good', 'great', 'thanks']
        negative_keywords = ['差', '坏', '问题', 'bad', 'problem', 'error']
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in user_message.lower())
        neg_count = sum(1 for keyword in negative_keywords if keyword in user_message.lower())
        
        if pos_count > neg_count:
            analysis['sentiment'] = 0.3
        elif neg_count > pos_count:
            analysis['sentiment'] = -0.3
        
        return analysis
    
    def _retrieve_relevant_memories(
        self, 
        user_message: str, 
        analysis: Dict[str, Any]
    ) -> List[EnhancedMemory]:
        """检索相关记忆"""
        # 基于内容搜索
        content_memories = self.evolution_system.search_enhanced_memories(
            user_message, limit=5
        )
        
        # 基于重要性获取记忆
        important_memories = []
        try:
            # 这里需要实现从数据库获取重要记忆的逻辑
            # 暂时返回空列表
            pass
        except:
            pass
        
        # 合并并去重
        all_memories = content_memories + important_memories
        unique_memories = []
        seen_ids = set()
        
        for memory in all_memories:
            if memory.id and memory.id not in seen_ids:
                unique_memories.append(memory)
                seen_ids.add(memory.id)
        
        return unique_memories[:8]  # 限制记忆数量
    
    async def _build_enhanced_message_history(
        self,
        user_message: str,
        analysis: Dict[str, Any],
        relevant_memories: List[EnhancedMemory]
    ) -> List[Dict[str, str]]:
        """构建增强消息历史"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加相关记忆作为上下文
        if relevant_memories:
            memory_context = "相关记忆和经验：\n"
            for memory in relevant_memories:
                importance_indicator = "🔥" if memory.importance > 0.8 else "⭐" if memory.importance > 0.6 else "💡"
                memory_context += f"{importance_indicator} {memory.content[:100]}...\n"
            
            messages.append({
                "role": "system",
                "content": memory_context
            })
        
        # 添加最近的对话历史（从基础记忆管理器）
        try:
            recent_conversations = self.memory_manager.get_recent_memories(
                limit=6, memory_type="conversation"
            )
            
            for conv in reversed(recent_conversations):
                if conv.metadata and "user_message" in conv.metadata:
                    messages.append({
                        "role": "user", 
                        "content": conv.metadata["user_message"]
                    })
                    messages.append({
                        "role": "assistant",
                        "content": conv.content
                    })
        except Exception as e:
            logger.warning(f"获取对话历史失败: {e}")
        
        # 添加消息分析信息
        if analysis['urgency'] > 0.7:
            messages.append({
                "role": "system",
                "content": "注意：用户的消息表现出较高的紧急程度，请优先快速、准确地回应。"
            })
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _standard_response_enhanced(
        self,
        messages: List[Dict[str, str]],
        user_message: str,
        analysis: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """增强标准响应处理"""
        # 调用OpenAI API
        response = await openai_client.chat_completion(messages=messages)
        
        # 记录交互到两个记忆系统
        await self._record_interaction_enhanced(user_message, response, analysis)
        
        # 评估表现并可能触发进化
        await self._evaluate_and_evolve_enhanced(user_message, response, analysis, start_time)
        
        # 添加增强信息
        response['analysis'] = analysis
        response['evolution_info'] = self._get_evolution_info()
        
        return response
    
    async def _stream_response_enhanced(
        self,
        messages: List[Dict[str, str]],
        user_message: str,
        analysis: Dict[str, Any],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """增强流式响应处理"""
        full_response = ""
        
        async for chunk in openai_client.stream_chat_completion(messages=messages):
            full_response += chunk
            yield chunk
        
        # 构建完整响应对象
        response = {
            "content": full_response,
            "model": config.openai_config.model,
            "request_time": time.time() - start_time,
            "stream": True,
            "analysis": analysis
        }
        
        # 记录交互
        await self._record_interaction_enhanced(user_message, response, analysis)
        
        # 评估表现
        await self._evaluate_and_evolve_enhanced(user_message, response, analysis, start_time)
    
    async def _record_interaction_enhanced(
        self,
        user_message: str,
        response: Dict[str, Any],
        analysis: Dict[str, Any]
    ):
        """增强交互记录"""
        # 记录到基础记忆系统
        basic_memory = Memory(
            content=response.get("content", ""),
            memory_type="conversation",
            importance=0.5,
            tags=["conversation", "interaction"],
            metadata={
                "user_message": user_message,
                "response_time": response.get("request_time", 0),
                "model_used": response.get("model", "unknown"),
                "error": response.get("error", False)
            }
        )
        self.memory_manager.add_memory(basic_memory)
        
        # 记录到增强记忆系统
        importance = self._calculate_interaction_importance(user_message, response, analysis)
        emotional_valence = analysis.get('sentiment', 0.0)
        
        enhanced_memory = EnhancedMemory(
            content=response.get("content", ""),
            memory_type="conversation",
            importance=importance,
            emotional_valence=emotional_valence,
            tags=["conversation", "interaction", f"intent_{analysis.get('intent', 'general')}"],
            metadata={
                "user_message": user_message,
                "response_time": response.get("request_time", 0),
                "analysis": analysis,
                "model_used": response.get("model", "unknown"),
                "error": response.get("error", False),
                "session_timestamp": time.time()
            }
        )
        
        self.evolution_system.add_enhanced_memory(enhanced_memory)
    
    def _calculate_interaction_importance(
        self,
        user_message: str,
        response: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> float:
        """计算交互重要性"""
        importance = 0.5  # 基础重要性
        
        # 基于消息复杂度
        importance += min(analysis.get('complexity', 0) * 0.1, 0.2)
        
        # 基于紧急程度
        importance += analysis.get('urgency', 0) * 0.2
        
        # 基于响应质量（基于响应时间和长度的简化评估）
        response_time = response.get("request_time", 5)
        if response_time < 2:
            importance += 0.1
        
        response_length = len(response.get("content", ""))
        if response_length > 200:
            importance += 0.1
        
        # 基于错误状态
        if response.get("error", False):
            importance += 0.2  # 错误交互也很重要，需要学习
        
        return min(importance, 1.0)
    
    async def _evaluate_and_evolve_enhanced(
        self,
        user_message: str,
        response: Dict[str, Any],
        analysis: Dict[str, Any],
        start_time: float
    ):
        """增强评估和进化"""
        # 构建详细的交互数据
        interaction_data = {
            "response_time": response.get("request_time", 0),
            "task_completed": not response.get("error", False),
            "error_count": 1 if response.get("error", False) else 0,
            "user_message_length": len(user_message),
            "response_length": len(response.get("content", "")),
            "complexity": analysis.get('complexity', 0),
            "urgency": analysis.get('urgency', 0),
            "sentiment_handled": abs(analysis.get('sentiment', 0)),
            "context_used": len(analysis.get('topics', [])),
            "memory_accessed": 1  # 简化版本
        }
        
        # 评估表现（使用多个指标）
        performance_scores = self._evaluate_multi_dimensional_performance(interaction_data)
        
        # 更新性能窗口
        overall_score = sum(performance_scores.values()) / len(performance_scores)
        self.evolution_system.performance_window.append({
            'score': overall_score,
            'detailed_scores': performance_scores,
            'timestamp': time.time()
        })
        
        # 检查是否需要进化
        if self._should_evolve_enhanced():
            evolution_result = self.evolution_system.execute_unified_evolution()
            
            logger.info(f"🧬 执行统一进化 - 策略: {evolution_result['strategy']}")
            logger.info(f"   执行时间: {evolution_result['execution_time']:.3f}秒")
            
            # 更新系统提示词
            self.system_prompt = self._generate_dynamic_system_prompt()
            
            # 记录进化事件
            evolution_memory = EnhancedMemory(
                content=f"执行统一进化 - 策略: {evolution_result['strategy']}",
                memory_type="evolution",
                importance=0.9,
                emotional_valence=0.5,
                tags=["evolution", "system_update", evolution_result['strategy']],
                metadata={
                    "evolution_result": evolution_result,
                    "trigger_scores": performance_scores
                }
            )
            self.evolution_system.add_enhanced_memory(evolution_memory)
    
    def _evaluate_multi_dimensional_performance(self, interaction_data: Dict[str, Any]) -> Dict[str, float]:
        """多维度性能评估"""
        scores = {}
        
        # 响应速度评分
        response_time = interaction_data.get("response_time", 5)
        if response_time < 1:
            scores['speed'] = 1.0
        elif response_time < 3:
            scores['speed'] = 0.8
        elif response_time < 6:
            scores['speed'] = 0.6
        else:
            scores['speed'] = 0.3
        
        # 任务完成评分
        scores['completion'] = 1.0 if interaction_data.get("task_completed", False) else 0.2
        
        # 错误处理评分
        error_count = interaction_data.get("error_count", 0)
        scores['reliability'] = max(0, 1.0 - error_count * 0.3)
        
        # 复杂度处理评分
        complexity = interaction_data.get("complexity", 0)
        response_length = interaction_data.get("response_length", 0)
        
        if complexity > 0:
            complexity_handling = min(response_length / (complexity * 100), 1.0)
            scores['complexity_handling'] = complexity_handling
        else:
            scores['complexity_handling'] = 0.8
        
        # 上下文利用评分
        context_score = min(interaction_data.get("context_used", 0) * 0.2, 1.0)
        scores['context_utilization'] = context_score
        
        return scores
    
    def _should_evolve_enhanced(self) -> bool:
        """增强进化判断"""
        if len(self.evolution_system.performance_window) < 30:
            return False
        
        recent_data = list(self.evolution_system.performance_window)[-30:]
        
        # 计算平均性能
        recent_scores = [item['score'] for item in recent_data]
        avg_performance = sum(recent_scores) / len(recent_scores)
        
        # 计算性能变化趋势
        mid_point = len(recent_scores) // 2
        early_avg = sum(recent_scores[:mid_point]) / mid_point
        recent_avg = sum(recent_scores[mid_point:]) / (len(recent_scores) - mid_point)
        performance_trend = recent_avg - early_avg
        
        # 进化触发条件
        trigger_conditions = [
            avg_performance < 0.6,  # 平均性能低
            performance_trend < -0.1,  # 性能下降
            len(self.evolution_system.evolution_history) == 0,  # 首次进化
            len(self.evolution_system.performance_window) % 100 == 0  # 定期进化
        ]
        
        return any(trigger_conditions)
    
    async def _handle_error_response(
        self,
        user_message: str,
        error_message: str,
        start_time: float
    ) -> Dict[str, Any]:
        """处理错误响应"""
        error_response = {
            "content": f"抱歉，我遇到了一个问题：{error_message}。我正在学习如何更好地处理这类情况。",
            "error": True,
            "error_message": error_message,
            "request_time": time.time() - start_time,
            "recovery_attempted": True
        }
        
        # 记录错误到增强记忆
        error_memory = EnhancedMemory(
            content=f"处理错误: {error_message}",
            memory_type="error",
            importance=0.8,
            emotional_valence=-0.3,
            tags=["error", "learning", "recovery"],
            metadata={
                "user_message": user_message,
                "error_details": error_message,
                "timestamp": time.time()
            }
        )
        self.evolution_system.add_enhanced_memory(error_memory)
        
        return error_response
    
    def _get_evolution_info(self) -> Dict[str, Any]:
        """获取进化信息"""
        status = self.evolution_system.get_system_status()
        
        return {
            "evolution_count": status.get("evolution_count", 0),
            "current_strategy_weights": status.get("strategy_weights", {}),
            "population_size": status.get("population_size", 0),
            "swarm_size": status.get("swarm_size", 0),
            "performance_window_size": status.get("performance_window_size", 0),
            "version": self.version
        }
    
    # === 公共接口方法 ===
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """获取增强状态信息"""
        basic_status = {
            "agent": {
                "name": self.name,
                "version": self.version,
                "uptime": time.time(),
                "personality": self.personality
            },
            "evolution_system": self.evolution_system.get_system_status(),
            "config": {
                "model": config.openai_config.model,
                "base_url": config.openai_config.base_url,
                "max_tokens": config.openai_config.max_tokens,
                "temperature": config.openai_config.temperature
            }
        }
        
        # 添加记忆统计
        try:
            basic_status["memory"] = self.memory_manager.get_memory_stats()
        except Exception as e:
            logger.warning(f"获取记忆统计失败: {e}")
            basic_status["memory"] = {"error": str(e)}
        
        return basic_status
    
    def trigger_manual_evolution(self) -> Dict[str, Any]:
        """手动触发进化"""
        logger.info("手动触发统一进化...")
        evolution_result = self.evolution_system.execute_unified_evolution()
        
        # 更新系统提示词
        self.system_prompt = self._generate_dynamic_system_prompt()
        
        return evolution_result
    
    def export_enhanced_data(self, filepath: str):
        """导出增强数据"""
        export_data = {
            "agent_info": {
                "name": self.name,
                "version": self.version,
                "personality": self.personality,
                "export_timestamp": time.time()
            },
            "evolution_system_status": self.evolution_system.get_system_status(),
            "performance_history": list(self.evolution_system.performance_window),
            "evolution_history": self.evolution_system.evolution_history[-50:]  # 最近50次进化
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"增强数据已导出到 {filepath}")
    
    async def test_all_systems(self) -> Dict[str, bool]:
        """测试所有系统"""
        test_results = {}
        
        # 测试OpenAI连接
        try:
            test_results["openai_connection"] = await openai_client.test_connection()
        except Exception as e:
            logger.error(f"OpenAI连接测试失败: {e}")
            test_results["openai_connection"] = False
        
        # 测试基础记忆系统
        try:
            test_memory = Memory(content="测试记忆", memory_type="test")
            memory_id = self.memory_manager.add_memory(test_memory)
            test_results["basic_memory"] = memory_id > 0
        except Exception as e:
            logger.error(f"基础记忆系统测试失败: {e}")
            test_results["basic_memory"] = False
        
        # 测试增强记忆系统
        try:
            test_enhanced_memory = EnhancedMemory(content="测试增强记忆", memory_type="test")
            enhanced_id = self.evolution_system.add_enhanced_memory(test_enhanced_memory)
            test_results["enhanced_memory"] = enhanced_id > 0
        except Exception as e:
            logger.error(f"增强记忆系统测试失败: {e}")
            test_results["enhanced_memory"] = False
        
        # 测试进化系统
        try:
            self.evolution_system.initialize_population(size=5)
            self.evolution_system.initialize_swarm(dimensions=5)
            test_results["evolution_system"] = True
        except Exception as e:
            logger.error(f"进化系统测试失败: {e}")
            test_results["evolution_system"] = False
        
        return test_results
    
    def cleanup_enhanced(self):
        """增强清理"""
        # 清理基础记忆
        self.memory_manager.cleanup_old_memories()
        
        # 清理增强记忆和应用遗忘曲线
        self.evolution_system.consolidate_memories()
        self.evolution_system.apply_forgetting_curve()
        
        # 保存进化数据
        self.evolution_system.save_evolution_data()
        
        logger.info("增强清理完成")

# 全局实例
unified_agent = UnifiedSelfEvolvingAgent()