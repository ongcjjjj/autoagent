"""
最小化配置管理模块
不依赖外部库，仅使用Python标准库
"""
import os
import json
from typing import Dict, Any, Optional

class OpenAIConfig:
    """OpenAI API配置"""
    
    def __init__(self, 
                 api_key: str = "",
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 2000,
                 temperature: float = 0.7,
                 timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenAIConfig':
        """从字典创建"""
        return cls(**data)

class AgentConfig:
    """Agent配置"""
    
    def __init__(self,
                 name: str = "SelfEvolvingAgent",
                 version: str = "1.0.0",
                 memory_limit: int = 1000,
                 learning_rate: float = 0.1,
                 evolution_threshold: int = 10,
                 auto_save: bool = True,
                 log_level: str = "INFO"):
        self.name = name
        self.version = version
        self.memory_limit = memory_limit
        self.learning_rate = learning_rate
        self.evolution_threshold = evolution_threshold
        self.auto_save = auto_save
        self.log_level = log_level
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "memory_limit": self.memory_limit,
            "learning_rate": self.learning_rate,
            "evolution_threshold": self.evolution_threshold,
            "auto_save": self.auto_save,
            "log_level": self.log_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """从字典创建"""
        return cls(**data)

class Config:
    """全局配置管理器"""
    
    def __init__(self, config_file: str = "agent_config.json"):
        self.config_file = config_file
        self.openai_config = self._load_openai_config()
        self.agent_config = self._load_agent_config()
    
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
                    return AgentConfig.from_dict(agent_data)
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
            "openai": self.openai_config.to_dict(),
            "agent": self.agent_config.to_dict()
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