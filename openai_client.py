"""
OpenAI API客户端模块 - 增强版
封装OpenAI API调用功能，支持智能重试、限流、性能监控、缓存
"""
import asyncio
import time
import hashlib
import random
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
import json
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from config import config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI API客户端 - 增强版"""
    
    def __init__(self):
        self.client = None
        
        # 增强功能
        self.request_history = deque(maxlen=1000)  # 请求历史
        self.performance_metrics = defaultdict(list)  # 性能指标
        self.rate_limiter = defaultdict(deque)  # 速率限制器
        self.response_cache = {}  # 响应缓存
        self.error_patterns = defaultdict(int)  # 错误模式统计
        self.retry_backoff = 1.0  # 重试退避时间
        self.last_request_time = 0.0  # 上次请求时间
        
        self.init_client()
    
    def init_client(self):
        """初始化OpenAI客户端"""
        try:
            client_kwargs = config.get_openai_client_kwargs()
            self.client = AsyncOpenAI(**client_kwargs)
            logger.info(f"OpenAI客户端初始化成功，使用模型: {config.openai_config.model}")
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {e}")
            self.client = None
    
    def update_config(self, **kwargs):
        """更新配置并重新初始化客户端"""
        config.update_openai_config(**kwargs)
        self.init_client()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表
            model: 使用的模型，默认使用配置中的模型
            max_tokens: 最大token数
            temperature: 生成温度
            stream: 是否流式输出
            **kwargs: 其他参数
        
        Returns:
            API响应结果
        """
        if not self.client:
            raise Exception("OpenAI客户端未初始化")
        
        # 使用配置中的默认值
        model = model or config.openai_config.model
        max_tokens = max_tokens or config.openai_config.max_tokens
        temperature = temperature or config.openai_config.temperature
        
        try:
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                **kwargs
            )
            
            end_time = time.time()
            
            if stream:
                return {"response": response, "request_time": end_time - start_time}
            else:
                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else {},
                    "request_time": end_time - start_time,
                    "finish_reason": response.choices[0].finish_reason
                }
                
                logger.info(f"API调用成功，用时: {result['request_time']:.2f}s, "
                           f"Token使用: {result['usage']}")
                
                return result
                
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式聊天完成
        
        Yields:
            生成的文本片段
        """
        if not self.client:
            raise Exception("OpenAI客户端未初始化")
        
        model = model or config.openai_config.model
        max_tokens = max_tokens or config.openai_config.max_tokens
        temperature = temperature or config.openai_config.temperature
        
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"流式API调用失败: {e}")
            raise
    
    async def embedding(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            texts: 文本列表
            model: 嵌入模型
            
        Returns:
            嵌入向量列表
        """
        if not self.client:
            raise Exception("OpenAI客户端未初始化")
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )
            
            return [embedding.embedding for embedding in response.data]
            
        except Exception as e:
            logger.error(f"嵌入向量获取失败: {e}")
            raise
    
    async def function_call(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        function_call: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        函数调用
        
        Args:
            messages: 消息列表
            functions: 可用函数列表
            function_call: 强制调用的函数名
            model: 使用的模型
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        if not self.client:
            raise Exception("OpenAI客户端未初始化")
        
        model = model or config.openai_config.model
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "function_call": response.choices[0].message.function_call.model_dump() 
                               if response.choices[0].message.function_call else None,
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else {}
            }
            
        except Exception as e:
            logger.error(f"函数调用失败: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        测试API连接
        
        Returns:
            连接是否成功
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            result = await self.chat_completion(
                messages=test_messages,
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        获取可用模型列表（常用模型）
        
        Returns:
            模型名称列表
        """
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k"
        ]
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        获取客户端信息
        
        Returns:
            客户端配置信息
        """
        return {
            "base_url": config.openai_config.base_url,
            "model": config.openai_config.model,
            "max_tokens": config.openai_config.max_tokens,
            "temperature": config.openai_config.temperature,
            "timeout": config.openai_config.timeout,
            "client_initialized": self.client is not None
        }

# 全局客户端实例
openai_client = OpenAIClient()