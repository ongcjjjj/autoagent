"""
配置管理模块 - 增强版
支持自定义OpenAI API配置、动态配置更新、配置验证和监控
"""
import os
import json
import logging
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from pathlib import Path

# 加载环境变量
load_dotenv()

# 设置日志
logger = logging.getLogger(__name__)

class OpenAIConfig(BaseModel):
    """OpenAI API配置 - 增强版"""
    api_key: str = Field(..., description="OpenAI API密钥")
    base_url: str = Field(default="https://api.openai.com/v1", description="API基础URL")
    model: str = Field(default="gpt-3.5-turbo", description="使用的模型")
    max_tokens: int = Field(default=2000, description="最大token数")
    temperature: float = Field(default=0.7, description="生成温度")
    timeout: int = Field(default=30, description="请求超时时间(秒)")
    retry_attempts: int = Field(default=3, description="重试次数")
    rate_limit_rpm: int = Field(default=60, description="每分钟请求限制")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v

class AgentConfig(BaseModel):
    """Agent配置 - 增强版"""
    name: str = Field(default="SelfEvolvingAgent", description="Agent名称")
    version: str = Field(default="3.0.0", description="Agent版本")
    memory_limit: int = Field(default=1000, description="记忆限制条数")
    learning_rate: float = Field(default=0.1, description="学习率")
    evolution_threshold: int = Field(default=10, description="进化阈值")
    auto_save: bool = Field(default=True, description="是否自动保存")
    log_level: str = Field(default="INFO", description="日志级别")
    performance_tracking: bool = Field(default=True, description="性能跟踪")
    auto_evolution: bool = Field(default=True, description="自动进化")
    context_window_size: int = Field(default=20, description="上下文窗口大小")
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if not 0 < v <= 1:
            raise ValueError('learning_rate must be between 0 and 1')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()

class Config:
    """全局配置管理器 - 增强版"""
    
    def __init__(self, config_file: str = "agent_config.json"):
        self.config_file = config_file
        self.config_history = []
        self.watchers = []
        self.last_modified = 0
        self.validation_errors = []
        
        # 创建配置目录
        self._ensure_config_directory()
        
        # 加载配置
        self.openai_config = self._load_openai_config()
        self.agent_config = self._load_agent_config()
        
        # 验证配置
        self._validate_configuration()
        
        logger.info(f"配置管理器初始化完成: {self.config_file}")
    
    def _ensure_config_directory(self):
        """确保配置目录存在"""
        config_dir = Path(self.config_file).parent
        config_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_openai_config(self) -> OpenAIConfig:
        """加载OpenAI配置"""
        # 优先从环境变量加载
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # 如果环境变量不存在，尝试从配置文件加载
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    openai_data = config_data.get("openai", {})
                    
                    return OpenAIConfig(
                        api_key=openai_data.get("api_key", api_key),
                        base_url=openai_data.get("base_url", base_url),
                        model=openai_data.get("model", model),
                        max_tokens=openai_data.get("max_tokens", 2000),
                        temperature=openai_data.get("temperature", 0.7),
                        timeout=openai_data.get("timeout", 30)
                    )
            except Exception as e:
                print(f"加载配置文件失败: {e}")
        
        return OpenAIConfig(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    
    def _load_agent_config(self) -> AgentConfig:
        """加载Agent配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    agent_data = config_data.get("agent", {})
                    return AgentConfig(**agent_data)
            except Exception as e:
                print(f"加载Agent配置失败: {e}")
        
        return AgentConfig()
    
    def update_openai_config(self, **kwargs) -> None:
        """更新OpenAI配置"""
        for key, value in kwargs.items():
            if hasattr(self.openai_config, key):
                setattr(self.openai_config, key, value)
        self.save_config()
    
    def update_agent_config(self, **kwargs) -> None:
        """更新Agent配置"""
        for key, value in kwargs.items():
            if hasattr(self.agent_config, key):
                setattr(self.agent_config, key, value)
        self.save_config()
    
    def save_config(self) -> None:
        """保存配置到文件"""
        config_data = {
            "openai": self.openai_config.model_dump(),
            "agent": self.agent_config.model_dump()
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def get_openai_client_kwargs(self) -> Dict[str, Any]:
        """获取OpenAI客户端参数"""
        return {
            "api_key": self.openai_config.api_key,
            "base_url": self.openai_config.base_url,
            "timeout": self.openai_config.timeout
        }

# 全局配置实例
config = Config()