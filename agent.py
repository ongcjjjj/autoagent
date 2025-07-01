"""
自我进化Agent主类 - 增强版
集成所有功能模块，支持智能对话、情感理解、主动学习
"""
import asyncio
import time
import json
import logging
import statistics
import random
from typing import Dict, List, Any, Optional, AsyncGenerator, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from config import config
from memory import Memory, MemoryManager
from evolution import EvolutionEngine
from openai_client import openai_client

logger = logging.getLogger(__name__)

class SelfEvolvingAgent:
    """自我进化Agent - 增强版"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or config.agent_config.name
        self.version = config.agent_config.version
        self.memory_manager = MemoryManager()
        self.evolution_engine = EvolutionEngine(self.memory_manager)
        self.conversation_history = []
        self.current_context = {}
        self.system_prompt = self._generate_system_prompt()
        
        # 增强功能
        self.emotion_state = {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}
        self.learning_metrics = {"interactions": 0, "improvements": 0, "errors": 0}
        self.user_profiles = defaultdict(dict)
        self.conversation_patterns = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        self.skill_levels = defaultdict(float)
        self.proactive_suggestions = []
        
        # 加载个性化设置
        self.load_personality()
        
        print(f"🤖 {self.name} v{self.version} 已启动 - 增强模式")
        self._log_startup()
    
    def _generate_system_prompt(self) -> str:
        """生成系统提示词 - 增强版"""
        adaptation_rules = self.evolution_engine.get_adaptation_rules()
        
        # 情感状态描述
        emotion_desc = self._get_emotion_description()
        
        # 技能水平描述
        top_skills = sorted(self.skill_levels.items(), key=lambda x: x[1], reverse=True)[:3]
        skills_desc = ", ".join([f"{skill}({level:.1f})" for skill, level in top_skills]) if top_skills else "正在学习中"
        
        base_prompt = f"""你是{self.name}，一个具备高级自我进化能力的AI助手v{self.version}。

🧠 核心特性：
- 智能学习和记忆管理系统
- 情感理解和共情能力
- 自主进化和性能优化
- 个性化交互适应
- 主动学习和技能提升

📊 当前状态：
- 情感状态: {emotion_desc}
- 交互次数: {self.learning_metrics['interactions']}
- 技能专长: {skills_desc}
- 学习改进: {self.learning_metrics['improvements']}次

⚙️ 适应规则："""
        
        for rule, value in adaptation_rules.items():
            base_prompt += f"\n- {rule}: {value}"
        
        base_prompt += f"""

🎯 行为指导：
1. 以友好、智能且富有洞察力的方式回应
2. 根据用户历史和偏好提供个性化帮助
3. 展现情感理解和共情能力
4. 主动学习和改进响应质量
5. 在适当时机提供主动建议和洞察

请始终保持专业、有帮助且具有人性化的交流风格。
"""
        
        return base_prompt
    
    def _get_emotion_description(self) -> str:
        """获取情感状态描述"""
        valence = self.emotion_state["valence"]
        arousal = self.emotion_state["arousal"]
        
        if valence > 0.3:
            if arousal > 0.6:
                return "积极兴奋"
            else:
                return "平静愉悦"
        elif valence < -0.3:
            if arousal > 0.6:
                return "焦虑不安"
            else:
                return "沮丧低落"
        else:
            if arousal > 0.6:
                return "中性兴奋"
            else:
                return "平静中性"
    
    def load_personality(self):
        """加载个性化设置"""
        try:
            with open("personality.json", "r", encoding="utf-8") as f:
                personality_data = json.load(f)
                self.personality = personality_data
        except FileNotFoundError:
            # 默认个性设置
            self.personality = {
                "communication_style": "friendly",
                "formality_level": "casual",
                "humor_usage": "moderate",
                "detail_level": "balanced",
                "proactivity": "moderate"
            }
            self.save_personality()
    
    def save_personality(self):
        """保存个性化设置"""
        with open("personality.json", "w", encoding="utf-8") as f:
            json.dump(self.personality, f, indent=2, ensure_ascii=False)
    
    def _log_startup(self):
        """记录启动日志"""
        startup_memory = Memory(
            content=f"Agent {self.name} 启动，版本: {self.version}",
            memory_type="system",
            importance=0.6,
            tags=["startup", "system"],
            metadata={
                "version": self.version,
                "timestamp": time.time()
            }
        )
        self.memory_manager.add_memory(startup_memory)
    
    async def process_message(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        user_id: Optional[str] = None
    ):
        """
        处理用户消息 - 增强版
        
        Args:
            user_message: 用户消息
            context: 额外上下文
            stream: 是否流式输出
            user_id: 用户ID（用于个性化）
            
        Returns:
            处理结果
        """
        start_time = time.time()
        self.learning_metrics["interactions"] += 1
        
        # 智能消息分析
        message_analysis = self._analyze_message_intelligence(user_message, user_id)
        
        # 更新情感状态
        self._update_emotion_state(message_analysis)
        
        # 更新上下文
        if context:
            self.current_context.update(context)
        
        # 记录对话模式
        self.conversation_patterns.append({
            "timestamp": start_time,
            "message_length": len(user_message),
            "sentiment": message_analysis.get("sentiment", 0),
            "complexity": message_analysis.get("complexity", 0)
        })
        
        # 构建增强消息历史
        messages = await self._build_enhanced_message_history(user_message, message_analysis, user_id)
        
        try:
            # 调用OpenAI API
            if stream:
                return self._stream_response_enhanced(messages, user_message, message_analysis, start_time)
            else:
                return await self._standard_response_enhanced(messages, user_message, message_analysis, start_time)
                
        except Exception as e:
            self.learning_metrics["errors"] += 1
            error_response = {
                "content": f"抱歉，我遇到了一个问题：{str(e)}。我正在学习如何更好地处理这类情况。",
                "error": True,
                "error_message": str(e),
                "request_time": time.time() - start_time,
                "analysis": message_analysis,
                "recovery_suggestions": self._generate_recovery_suggestions(str(e))
            }
            
            # 记录错误用于学习
            await self._record_interaction_enhanced(user_message, error_response, message_analysis)
            return error_response
    
    def _analyze_message_intelligence(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """智能消息分析"""
        analysis = {
            "length": len(message),
            "word_count": len(message.split()),
            "complexity": self._calculate_complexity(message),
            "sentiment": self._analyze_sentiment(message),
            "intent": self._detect_intent(message),
            "topics": self._extract_topics(message),
            "urgency": self._assess_urgency(message),
            "formality": self._assess_formality(message),
            "user_id": user_id
        }
        
        # 用户个性化分析
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            analysis["user_familiarity"] = profile.get("interaction_count", 0)
            analysis["preferred_style"] = profile.get("preferred_style", "balanced")
        
        return analysis
    
    def _calculate_complexity(self, message: str) -> float:
        """计算消息复杂度"""
        words = message.split()
        if not words:
            return 0.0
        
        # 基于词汇长度、句子数量和特殊词汇
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = message.count('.') + message.count('!') + message.count('?') + 1
        technical_words = sum(1 for word in words if len(word) > 8)
        
        complexity = (avg_word_length / 10 + sentence_count / 10 + technical_words / len(words))
        return min(complexity, 1.0)
    
    def _analyze_sentiment(self, message: str) -> float:
        """分析情感倾向"""
        positive_words = ['好', '棒', '谢谢', '喜欢', '满意', 'good', 'great', 'thanks', 'love', 'excellent']
        negative_words = ['差', '坏', '问题', '困难', '不满', 'bad', 'poor', 'problem', 'difficult', 'hate']
        
        words = message.lower().split()
        positive_count = sum(1 for word in words if any(pos in word for pos in positive_words))
        negative_count = sum(1 for word in words if any(neg in word for neg in negative_words))
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return max(-1.0, min(1.0, sentiment))
    
    def _detect_intent(self, message: str) -> str:
        """检测用户意图"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['帮助', '怎么', '如何', 'help', 'how']):
            return "help_request"
        elif any(word in message_lower for word in ['解释', '说明', 'explain', 'what']):
            return "explanation"
        elif any(word in message_lower for word in ['创建', '生成', '写', 'create', 'generate']):
            return "creation"
        elif any(word in message_lower for word in ['分析', '评估', 'analyze', 'evaluate']):
            return "analysis"
        elif any(word in message_lower for word in ['你好', '嗨', 'hello', 'hi']):
            return "greeting"
        else:
            return "general"
    
    def _extract_topics(self, message: str) -> List[str]:
        """提取主题关键词"""
        topic_keywords = {
            "技术": ["编程", "代码", "算法", "数据", "programming", "code", "algorithm"],
            "学习": ["学习", "教学", "知识", "学会", "learning", "study", "knowledge"],
            "工作": ["工作", "项目", "任务", "业务", "work", "project", "task"],
            "生活": ["生活", "日常", "健康", "食物", "life", "daily", "health"]
        }
        
        message_lower = message.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    async def _build_message_history(self, user_message: str) -> List[Dict[str, str]]:
        """构建消息历史"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加重要记忆作为上下文
        important_memories = self.memory_manager.get_important_memories(limit=5)
        if important_memories:
            context_content = "相关记忆：\n"
            for memory in important_memories:
                context_content += f"- {memory.content}\n"
            
            messages.append({
                "role": "system",
                "content": context_content
            })
        
        # 添加最近的对话历史
        recent_conversations = self.memory_manager.get_recent_memories(
            limit=10,
            memory_type="conversation"
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
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _standard_response(
        self,
        messages: List[Dict[str, str]],
        user_message: str,
        start_time: float
    ) -> Dict[str, Any]:
        """标准响应处理"""
        response = await openai_client.chat_completion(messages=messages)
        
        # 记录交互
        await self._record_interaction(user_message, response)
        
        # 评估表现并可能触发进化
        await self._evaluate_and_evolve(user_message, response, start_time)
        
        return response
    
    async def _stream_response(
        self,
        messages: List[Dict[str, str]],
        user_message: str,
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """流式响应处理"""
        full_response = ""
        
        async for chunk in openai_client.stream_chat_completion(messages=messages):
            full_response += chunk
            yield chunk
        
        # 构建完整响应对象用于记录
        response = {
            "content": full_response,
            "model": config.openai_config.model,
            "request_time": time.time() - start_time,
            "stream": True
        }
        
        # 记录交互
        await self._record_interaction(user_message, response)
        
        # 评估表现
        await self._evaluate_and_evolve(user_message, response, start_time)
    
    async def _record_interaction(self, user_message: str, response: Dict[str, Any]):
        """记录交互到记忆中"""
        interaction_memory = Memory(
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
        
        self.memory_manager.add_memory(interaction_memory)
    
    async def _evaluate_and_evolve(
        self,
        user_message: str,
        response: Dict[str, Any],
        start_time: float
    ):
        """评估表现并可能触发进化"""
        # 构建交互数据
        interaction_data = {
            "response_time": response.get("request_time", 0),
            "task_completed": not response.get("error", False),
            "error_count": 1 if response.get("error", False) else 0,
            "user_message_length": len(user_message),
            "response_length": len(response.get("content", ""))
        }
        
        # 评估表现
        performance_score = self.evolution_engine.evaluate_performance(interaction_data)
        self.evolution_engine.update_performance_window(performance_score)
        
        # 检查是否需要进化
        if self.evolution_engine.should_evolve():
            evolution_record = self.evolution_engine.execute_evolution()
            print(f"🧬 执行进化 {evolution_record.version}")
            print(f"   改进领域: {', '.join(evolution_record.improvement_areas)}")
            
            # 更新系统提示词
            self.system_prompt = self._generate_system_prompt()
    
    def update_config(self, **kwargs):
        """更新配置"""
        openai_client.update_config(**kwargs)
        print("配置已更新")
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        memory_stats = self.memory_manager.get_memory_stats()
        evolution_summary = self.evolution_engine.get_evolution_summary()
        client_info = openai_client.get_client_info()
        
        return {
            "agent": {
                "name": self.name,
                "version": self.version,
                "uptime": time.time(),
                "personality": self.personality
            },
            "memory": memory_stats,
            "evolution": evolution_summary,
            "openai_client": client_info,
            "config": {
                "model": config.openai_config.model,
                "base_url": config.openai_config.base_url,
                "max_tokens": config.openai_config.max_tokens,
                "temperature": config.openai_config.temperature
            }
        }
    
    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索记忆"""
        memories = self.memory_manager.search_memories(query, limit)
        return [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type,
                "importance": memory.importance,
                "timestamp": memory.timestamp,
                "tags": memory.tags
            }
            for memory in memories
        ]
    
    def add_manual_memory(
        self,
        content: str,
        memory_type: str = "knowledge",
        importance: float = 0.7,
        tags: Optional[List[str]] = None
    ) -> int:
        """手动添加记忆"""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            metadata={"source": "manual"}
        )
        
        return self.memory_manager.add_memory(memory)
    
    def export_data(self, filepath: str):
        """导出数据"""
        export_data = {
            "agent_info": {
                "name": self.name,
                "version": self.version,
                "personality": self.personality
            },
            "config": {
                "openai": config.openai_config.model_dump(),
                "agent": config.agent_config.model_dump()
            },
            "timestamp": time.time()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        # 导出记忆
        memory_filepath = filepath.replace(".json", "_memories.json")
        self.memory_manager.export_memories(memory_filepath)
        
        print(f"数据已导出到 {filepath} 和 {memory_filepath}")
    
    def cleanup(self):
        """清理资源"""
        self.memory_manager.cleanup_old_memories()
        print("资源清理完成")
    
    async def test_connection(self) -> bool:
        """测试API连接"""
        return await openai_client.test_connection()
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """获取进化历史"""
        return [
            {
                "version": record.version,
                "timestamp": record.timestamp,
                "improvements": record.improvement_areas,
                "strategies": record.changes,
                "metrics": {
                    "success_rate": record.metrics.success_rate,
                    "response_quality": record.metrics.response_quality,
                    "learning_efficiency": record.metrics.learning_efficiency
                }
            }
            for record in self.evolution_engine.evolution_history
        ]