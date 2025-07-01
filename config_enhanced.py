"""
配置管理模块 - 深度增强版
在原有基础上增加：动态配置热更新、配置历史追踪、智能配置建议、多环境支持、安全配置管理
"""
import os
import json
import logging
import time
import hashlib
import threading
import asyncio
from typing import Optional, Dict, Any, List, Callable, Union
from pydantic import BaseModel, Field, validator, SecretStr
from dotenv import load_dotenv
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import yaml

# 加载环境变量
load_dotenv()

# 设置日志
logger = logging.getLogger(__name__)

class ConfigEnvironment(Enum):
    """配置环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigSecurity(Enum):
    """配置安全级别"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    timestamp: float
    section: str
    key: str
    old_value: Any
    new_value: Any
    user: str = "system"
    reason: str = ""
    
class OpenAIConfigEnhanced(BaseModel):
    """OpenAI API配置 - 深度增强版"""
    api_key: SecretStr = Field(..., description="OpenAI API密钥")
    base_url: str = Field(default="https://api.openai.com/v1", description="API基础URL")
    model: str = Field(default="gpt-3.5-turbo", description="使用的模型")
    max_tokens: int = Field(default=2000, description="最大token数")
    temperature: float = Field(default=0.7, description="生成温度")
    timeout: int = Field(default=30, description="请求超时时间(秒)")
    retry_attempts: int = Field(default=3, description="重试次数")
    rate_limit_rpm: int = Field(default=60, description="每分钟请求限制")
    
    # 高级配置
    backup_models: List[str] = Field(default_factory=lambda: ["gpt-3.5-turbo-16k"], description="备用模型列表")
    auto_fallback: bool = Field(default=True, description="自动降级到备用模型")
    cost_threshold: float = Field(default=10.0, description="成本阈值(美元)")
    quality_threshold: float = Field(default=0.8, description="质量阈值")
    monitoring_enabled: bool = Field(default=True, description="监控启用")
    adaptive_timeout: bool = Field(default=True, description="自适应超时")
    compression_enabled: bool = Field(default=False, description="请求压缩")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v <= 0 or v > 32000:
            raise ValueError('max_tokens must be between 1 and 32000')
        return v
    
    @validator('rate_limit_rpm')
    def validate_rate_limit(cls, v):
        if v <= 0 or v > 10000:
            raise ValueError('rate_limit_rpm must be between 1 and 10000')
        return v

class AgentConfigEnhanced(BaseModel):
    """Agent配置 - 深度增强版"""
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
    
    # 深度增强配置
    personality_profile: str = Field(default="balanced", description="个性化配置文件")
    emotional_intelligence: bool = Field(default=True, description="情感智能")
    proactive_learning: bool = Field(default=True, description="主动学习")
    multi_language_support: bool = Field(default=True, description="多语言支持")
    creativity_level: float = Field(default=0.7, description="创造力水平")
    ethical_guidelines: bool = Field(default=True, description="伦理准则")
    safety_mode: bool = Field(default=True, description="安全模式")
    debug_mode: bool = Field(default=False, description="调试模式")
    
    # 性能配置
    max_concurrent_requests: int = Field(default=5, description="最大并发请求数")
    cache_enabled: bool = Field(default=True, description="缓存启用")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")
    optimization_level: str = Field(default="standard", description="优化级别")
    
    # 集成配置
    plugins_enabled: List[str] = Field(default_factory=list, description="启用的插件")
    external_apis: Dict[str, str] = Field(default_factory=dict, description="外部API配置")
    webhook_urls: List[str] = Field(default_factory=list, description="Webhook URLs")
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if not 0 < v <= 1:
            raise ValueError('learning_rate must be between 0 and 1')
        return v
    
    @validator('creativity_level')
    def validate_creativity_level(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('creativity_level must be between 0 and 1')
        return v
    
    @validator('optimization_level')
    def validate_optimization_level(cls, v):
        valid_levels = ['minimal', 'standard', 'aggressive', 'maximum']
        if v not in valid_levels:
            raise ValueError(f'optimization_level must be one of {valid_levels}')
        return v

class SecurityConfig(BaseModel):
    """安全配置"""
    encryption_enabled: bool = Field(default=True, description="加密启用")
    access_logging: bool = Field(default=True, description="访问日志")
    rate_limiting: bool = Field(default=True, description="速率限制")
    ip_whitelist: List[str] = Field(default_factory=list, description="IP白名单")
    api_key_rotation: bool = Field(default=False, description="API密钥轮换")
    audit_trail: bool = Field(default=True, description="审计跟踪")
    
class ConfigWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_manager: 'ConfigManagerEnhanced'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logger.info(f"配置文件变更检测: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_config())

class ConfigManagerEnhanced:
    """增强配置管理器 - 深度优化版"""
    
    def __init__(self, 
                 config_file: str = "agent_config.json",
                 environment: str = "development",
                 watch_changes: bool = True):
        self.config_file = config_file
        self.environment = ConfigEnvironment(environment)
        self.watch_changes = watch_changes
        
        # 配置历史和监控
        self.config_history: List[ConfigChangeEvent] = []
        self.change_callbacks: List[Callable] = []
        self.validation_errors: List[str] = []
        self.last_modified = 0
        self.config_hash = ""
        
        # 文件监控
        self.observer = None
        if watch_changes:
            self._setup_file_watcher()
        
        # 多环境配置
        self.environment_configs = {}
        
        # 创建配置目录结构
        self._ensure_config_structure()
        
        # 加载配置
        self.openai_config = self._load_openai_config()
        self.agent_config = self._load_agent_config()
        self.security_config = self._load_security_config()
        
        # 配置验证和优化
        self._validate_configuration()
        self._optimize_configuration()
        self._setup_auto_backup()
        
        logger.info(f"增强配置管理器初始化完成: 环境={self.environment.value}")
    
    def _ensure_config_structure(self):
        """确保配置目录结构存在"""
        base_dir = Path(self.config_file).parent
        directories = [
            base_dir,
            base_dir / "environments",
            base_dir / "backups",
            base_dir / "schemas",
            base_dir / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_file_watcher(self):
        """设置文件监控"""
        try:
            self.observer = Observer()
            event_handler = ConfigWatcher(self)
            config_dir = Path(self.config_file).parent
            self.observer.schedule(event_handler, str(config_dir), recursive=True)
            self.observer.start()
            logger.info("配置文件监控已启动")
        except Exception as e:
            logger.warning(f"配置文件监控启动失败: {e}")
    
    def _load_openai_config(self) -> OpenAIConfigEnhanced:
        """加载OpenAI配置 - 增强版"""
        # 环境变量优先级最高
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # 加载环境特定配置
        env_config = self._load_environment_config()
        
        # 尝试从配置文件加载
        config_data = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        openai_data = config_data.get("openai", {})
        
        # 合并配置（环境变量 > 环境配置 > 文件配置 > 默认值）
        final_config = {
            "api_key": api_key or env_config.get("openai", {}).get("api_key", openai_data.get("api_key", "")),
            "base_url": base_url or env_config.get("openai", {}).get("base_url", openai_data.get("base_url", "https://api.openai.com/v1")),
            "model": model or env_config.get("openai", {}).get("model", openai_data.get("model", "gpt-3.5-turbo")),
            **{k: v for k, v in openai_data.items() if k not in ["api_key", "base_url", "model"]}
        }
        
        return OpenAIConfigEnhanced(**final_config)
    
    def _load_agent_config(self) -> AgentConfigEnhanced:
        """加载Agent配置 - 增强版"""
        env_config = self._load_environment_config()
        
        config_data = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.error(f"加载Agent配置失败: {e}")
        
        agent_data = config_data.get("agent", {})
        env_agent_data = env_config.get("agent", {})
        
        # 合并配置
        final_config = {**agent_data, **env_agent_data}
        
        return AgentConfigEnhanced(**final_config)
    
    def _load_security_config(self) -> SecurityConfig:
        """加载安全配置"""
        config_data = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.error(f"加载安全配置失败: {e}")
        
        security_data = config_data.get("security", {})
        return SecurityConfig(**security_data)
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """加载环境特定配置"""
        env_file = Path(self.config_file).parent / "environments" / f"{self.environment.value}.json"
        
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载环境配置失败: {e}")
        
        return {}
    
    def _validate_configuration(self):
        """验证配置完整性和正确性"""
        errors = []
        
        # 验证必要字段
        if not self.openai_config.api_key.get_secret_value():
            errors.append("OpenAI API密钥未设置")
        
        # 验证模型可用性
        if self.openai_config.model not in self._get_supported_models():
            errors.append(f"不支持的模型: {self.openai_config.model}")
        
        # 验证性能配置
        if self.agent_config.max_concurrent_requests > 20:
            errors.append("并发请求数过高，可能影响性能")
        
        # 验证安全配置
        if self.environment == ConfigEnvironment.PRODUCTION and not self.security_config.encryption_enabled:
            errors.append("生产环境必须启用加密")
        
        self.validation_errors = errors
        
        if errors:
            logger.warning(f"配置验证发现 {len(errors)} 个问题:")
            for error in errors:
                logger.warning(f"  - {error}")
    
    def _optimize_configuration(self):
        """智能配置优化"""
        optimizations = []
        
        # 根据环境优化
        if self.environment == ConfigEnvironment.PRODUCTION:
            if self.agent_config.debug_mode:
                self.agent_config.debug_mode = False
                optimizations.append("生产环境自动关闭调试模式")
        
        # 性能优化建议
        if self.agent_config.cache_enabled and self.agent_config.cache_ttl < 300:
            optimizations.append("建议增加缓存TTL以提升性能")
        
        # 成本优化
        if self.openai_config.max_tokens > 4000 and self.openai_config.model == "gpt-4":
            optimizations.append("使用GPT-4时建议降低max_tokens以控制成本")
        
        if optimizations:
            logger.info("配置优化建议:")
            for opt in optimizations:
                logger.info(f"  - {opt}")
    
    def _setup_auto_backup(self):
        """设置自动备份"""
        backup_dir = Path(self.config_file).parent / "backups"
        backup_file = backup_dir / f"config_backup_{int(time.time())}.json"
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"配置备份已创建: {backup_file}")
        except Exception as e:
            logger.error(f"创建配置备份失败: {e}")
    
    def _get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            "gpt-4", "gpt-4-turbo-preview", "gpt-4-1106-preview",
            "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k"
        ]
    
    async def reload_config(self):
        """热重载配置"""
        try:
            old_openai = self.openai_config.model_dump()
            old_agent = self.agent_config.model_dump()
            
            # 重新加载配置
            self.openai_config = self._load_openai_config()
            self.agent_config = self._load_agent_config()
            self.security_config = self._load_security_config()
            
            # 重新验证
            self._validate_configuration()
            
            # 记录变更
            self._record_config_changes(old_openai, old_agent)
            
            # 触发回调
            await self._trigger_change_callbacks()
            
            logger.info("配置热重载完成")
            
        except Exception as e:
            logger.error(f"配置热重载失败: {e}")
    
    def _record_config_changes(self, old_openai: Dict, old_agent: Dict):
        """记录配置变更"""
        current_openai = self.openai_config.model_dump()
        current_agent = self.agent_config.model_dump()
        
        # 比较OpenAI配置变更
        for key, new_value in current_openai.items():
            old_value = old_openai.get(key)
            if old_value != new_value:
                change_event = ConfigChangeEvent(
                    timestamp=time.time(),
                    section="openai",
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    reason="hot_reload"
                )
                self.config_history.append(change_event)
        
        # 比较Agent配置变更
        for key, new_value in current_agent.items():
            old_value = old_agent.get(key)
            if old_value != new_value:
                change_event = ConfigChangeEvent(
                    timestamp=time.time(),
                    section="agent",
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    reason="hot_reload"
                )
                self.config_history.append(change_event)
    
    async def _trigger_change_callbacks(self):
        """触发配置变更回调"""
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"配置变更回调执行失败: {e}")
    
    def add_change_callback(self, callback: Callable):
        """添加配置变更回调"""
        self.change_callbacks.append(callback)
    
    def update_openai_config(self, **kwargs) -> None:
        """更新OpenAI配置 - 增强版"""
        old_values = {}
        
        for key, value in kwargs.items():
            if hasattr(self.openai_config, key):
                old_values[key] = getattr(self.openai_config, key)
                setattr(self.openai_config, key, value)
                
                # 记录变更
                change_event = ConfigChangeEvent(
                    timestamp=time.time(),
                    section="openai",
                    key=key,
                    old_value=old_values[key],
                    new_value=value,
                    reason="manual_update"
                )
                self.config_history.append(change_event)
        
        self.save_config()
        logger.info(f"OpenAI配置已更新: {list(kwargs.keys())}")
    
    def update_agent_config(self, **kwargs) -> None:
        """更新Agent配置 - 增强版"""
        old_values = {}
        
        for key, value in kwargs.items():
            if hasattr(self.agent_config, key):
                old_values[key] = getattr(self.agent_config, key)
                setattr(self.agent_config, key, value)
                
                # 记录变更
                change_event = ConfigChangeEvent(
                    timestamp=time.time(),
                    section="agent",
                    key=key,
                    old_value=old_values[key],
                    new_value=value,
                    reason="manual_update"
                )
                self.config_history.append(change_event)
        
        self.save_config()
        logger.info(f"Agent配置已更新: {list(kwargs.keys())}")
    
    def save_config(self) -> None:
        """保存配置到文件 - 增强版"""
        config_data = {
            "openai": self.openai_config.model_dump(),
            "agent": self.agent_config.model_dump(),
            "security": self.security_config.model_dump(),
            "metadata": {
                "last_modified": time.time(),
                "environment": self.environment.value,
                "config_version": "3.0.0"
            }
        }
        
        # 敏感信息处理
        if "api_key" in config_data["openai"]:
            config_data["openai"]["api_key"] = "***HIDDEN***"
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 更新配置哈希
            self.config_hash = self._calculate_config_hash(config_data)
            logger.info("配置已保存")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _calculate_config_hash(self, config_data: Dict) -> str:
        """计算配置哈希"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def export_config(self, filepath: str, include_sensitive: bool = False):
        """导出配置"""
        config_data = {
            "openai": self.openai_config.model_dump(),
            "agent": self.agent_config.model_dump(),
            "security": self.security_config.model_dump()
        }
        
        if not include_sensitive:
            # 移除敏感信息
            if "api_key" in config_data["openai"]:
                config_data["openai"]["api_key"] = "***HIDDEN***"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已导出到: {filepath}")
    
    def import_config(self, filepath: str, merge: bool = True):
        """导入配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if merge:
                # 合并模式
                if "openai" in import_data:
                    for key, value in import_data["openai"].items():
                        if hasattr(self.openai_config, key):
                            setattr(self.openai_config, key, value)
                
                if "agent" in import_data:
                    for key, value in import_data["agent"].items():
                        if hasattr(self.agent_config, key):
                            setattr(self.agent_config, key, value)
            else:
                # 替换模式
                if "openai" in import_data:
                    self.openai_config = OpenAIConfigEnhanced(**import_data["openai"])
                if "agent" in import_data:
                    self.agent_config = AgentConfigEnhanced(**import_data["agent"])
            
            self.save_config()
            logger.info(f"配置已从 {filepath} 导入")
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
    
    def get_config_suggestions(self) -> List[str]:
        """获取配置优化建议"""
        suggestions = []
        
        # 性能建议
        if self.agent_config.max_concurrent_requests < 3:
            suggestions.append("建议增加并发请求数以提升性能")
        
        if not self.agent_config.cache_enabled:
            suggestions.append("建议启用缓存以减少API调用")
        
        # 成本建议
        if self.openai_config.max_tokens > 2000 and self.openai_config.model.startswith("gpt-4"):
            suggestions.append("使用GPT-4时建议适当降低max_tokens以控制成本")
        
        # 安全建议
        if self.environment == ConfigEnvironment.PRODUCTION:
            if not self.security_config.rate_limiting:
                suggestions.append("生产环境建议启用速率限制")
            if not self.security_config.access_logging:
                suggestions.append("生产环境建议启用访问日志")
        
        return suggestions
    
    def get_config_history(self, limit: int = 50) -> List[ConfigChangeEvent]:
        """获取配置变更历史"""
        return self.config_history[-limit:]
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取综合配置状态"""
        return {
            "config_file": self.config_file,
            "environment": self.environment.value,
            "last_modified": self.last_modified,
            "config_hash": self.config_hash,
            "validation_errors": self.validation_errors,
            "change_count": len(self.config_history),
            "file_watcher_active": self.observer is not None and self.observer.is_alive(),
            "openai_config": {
                "model": self.openai_config.model,
                "max_tokens": self.openai_config.max_tokens,
                "temperature": self.openai_config.temperature,
                "api_key_set": bool(self.openai_config.api_key.get_secret_value())
            },
            "agent_config": {
                "name": self.agent_config.name,
                "version": self.agent_config.version,
                "performance_tracking": self.agent_config.performance_tracking,
                "auto_evolution": self.agent_config.auto_evolution
            },
            "security_config": {
                "encryption_enabled": self.security_config.encryption_enabled,
                "access_logging": self.security_config.access_logging,
                "rate_limiting": self.security_config.rate_limiting
            },
            "suggestions": self.get_config_suggestions()
        }
    
    def cleanup(self):
        """清理资源"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        logger.info("配置管理器已清理")

# 全局增强配置实例
config_enhanced = ConfigManagerEnhanced()