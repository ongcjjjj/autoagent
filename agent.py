"""
自我进化Agent主类
集成所有功能模块
"""
import asyncio
import time
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from config import config
from memory import Memory, MemoryManager
from evolution import EvolutionEngine
from openai_client import openai_client

class SelfEvolvingAgent:
    """自我进化Agent"""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or config.agent_config.name
        self.version = config.agent_config.version
        self.memory_manager = MemoryManager()
        self.evolution_engine = EvolutionEngine(self.memory_manager)
        self.conversation_history = []
        self.current_context = {}
        self.system_prompt = self._generate_system_prompt()
        
        # 加载个性化设置
        self.load_personality()
        
        print(f"🤖 {self.name} v{self.version} 已启动")
        self._log_startup()
    
    def _generate_system_prompt(self) -> str:
        """生成系统提示词"""
        adaptation_rules = self.evolution_engine.get_adaptation_rules()
        
        base_prompt = f"""你是{self.name}，一个具备自我进化能力的AI助手。

核心特性：
- 能够学习和记忆交互历史
- 根据表现自动优化和改进
- 具有情感理解和表达能力
- 能够适应不同的交互风格

当前适应规则：
"""
        
        for rule, value in adaptation_rules.items():
            base_prompt += f"- {rule}: {value}\n"
        
        base_prompt += """
请以友好、专业且有帮助的方式回应用户。根据上下文和历史记忆提供个性化的帮助。
"""
        
        return base_prompt
    
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
        stream: bool = False
    ):
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
            context: 额外上下文
            stream: 是否流式输出
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 更新上下文
        if context:
            self.current_context.update(context)
        
        # 构建消息历史
        messages = await self._build_message_history(user_message)
        
        try:
            # 调用OpenAI API
            if stream:
                # 对于流式响应，直接返回生成器
                return self._stream_response(messages, user_message, start_time)
            else:
                return await self._standard_response(messages, user_message, start_time)
                
        except Exception as e:
            error_response = {
                "content": f"抱歉，我遇到了一个问题：{str(e)}",
                "error": True,
                "error_message": str(e),
                "request_time": time.time() - start_time
            }
            
            # 记录错误
            await self._record_interaction(user_message, error_response)
            return error_response
    
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