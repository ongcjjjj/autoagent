"""
统一配置管理系统
整合所有模块的配置需求
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class OpenAIConfig:
    """OpenAI API配置"""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 30

@dataclass  
class AgentConfig:
    """Agent基础配置"""
    name: str = "IntegratedEvolutionAgent"
    version: str = "3.0.0"
    memory_limit: int = 1000
    learning_rate: float = 0.1
    evolution_threshold: int = 10
    auto_save: bool = True
    log_level: str = "INFO"

@dataclass
class EvolutionConfig:
    """进化系统配置"""
    population_size: int = 50
    swarm_size: int = 30
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2
    w_inertia: float = 0.7
    c1_cognitive: float = 1.4
    c2_social: float = 1.4
    performance_window_size: int = 1000
    evolution_history_limit: int = 100

@dataclass
class MemoryConfig:
    """记忆系统配置"""
    basic_db_path: str = "agent_memory.db"
    enhanced_db_path: str = "unified_evolution.db"
    consolidation_threshold: float = 0.7
    forgetting_rate: float = 0.1
    max_associations: int = 20
    cleanup_days: int = 30

@dataclass
class SystemConfig:
    """系统级配置"""
    data_directory: str = "data"
    log_directory: str = "logs"
    export_directory: str = "exports"
    backup_directory: str = "backups"
    max_log_files: int = 10
    backup_retention_days: int = 7
    auto_backup: bool = True

class IntegratedConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_file: str = "integrated_config.json"):
        self.config_file = config_file
        self.openai_config = OpenAIConfig()
        self.agent_config = AgentConfig()
        self.evolution_config = EvolutionConfig()
        self.memory_config = MemoryConfig()
        self.system_config = SystemConfig()
        
        # 创建必要目录
        self._create_directories()
        
        # 加载配置
        self.load_config()
        
        logger.info("✅ 统一配置管理器初始化完成")
    
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            self.system_config.data_directory,
            self.system_config.log_directory,
            self.system_config.export_directory,
            self.system_config.backup_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self):
        """加载配置"""
        try:
            # 优先从环境变量加载敏感配置
            self._load_from_env()
            
            # 从配置文件加载
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 更新各模块配置
                if "openai" in config_data:
                    self._update_config(self.openai_config, config_data["openai"])
                
                if "agent" in config_data:
                    self._update_config(self.agent_config, config_data["agent"])
                
                if "evolution" in config_data:
                    self._update_config(self.evolution_config, config_data["evolution"])
                
                if "memory" in config_data:
                    self._update_config(self.memory_config, config_data["memory"])
                
                if "system" in config_data:
                    self._update_config(self.system_config, config_data["system"])
                
                logger.info("📄 从配置文件加载配置完成")
            else:
                logger.info("📄 使用默认配置，将创建新的配置文件")
                self.save_config()
                
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            logger.info("使用默认配置")
    
    def _load_from_env(self):
        """从环境变量加载敏感配置"""
        # OpenAI配置
        if os.getenv("OPENAI_API_KEY"):
            self.openai_config.api_key = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_BASE_URL"):
            self.openai_config.base_url = os.getenv("OPENAI_BASE_URL")
        
        if os.getenv("OPENAI_MODEL"):
            self.openai_config.model = os.getenv("OPENAI_MODEL")
        
        # Agent配置
        if os.getenv("AGENT_NAME"):
            self.agent_config.name = os.getenv("AGENT_NAME")
        
        if os.getenv("LOG_LEVEL"):
            self.agent_config.log_level = os.getenv("LOG_LEVEL")
    
    def _update_config(self, config_obj, config_data: Dict[str, Any]):
        """更新配置对象"""
        for key, value in config_data.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def save_config(self):
        """保存配置到文件"""
        config_data = {
            "openai": asdict(self.openai_config),
            "agent": asdict(self.agent_config),
            "evolution": asdict(self.evolution_config),
            "memory": asdict(self.memory_config),
            "system": asdict(self.system_config),
            "meta": {
                "version": "3.0.0",
                "last_updated": self._get_timestamp(),
                "description": "统一自主进化Agent系统配置"
            }
        }
        
        # 隐藏敏感信息
        if config_data["openai"]["api_key"]:
            config_data["openai"]["api_key"] = "***HIDDEN***"
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info("💾 配置保存成功")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_openai_client_kwargs(self) -> Dict[str, Any]:
        """获取OpenAI客户端参数"""
        return {
            "api_key": self.openai_config.api_key,
            "base_url": self.openai_config.base_url,
            "timeout": self.openai_config.timeout
        }
    
    def get_database_paths(self) -> Dict[str, str]:
        """获取数据库路径"""
        data_dir = self.system_config.data_directory
        return {
            "basic_memory": os.path.join(data_dir, self.memory_config.basic_db_path),
            "enhanced_memory": os.path.join(data_dir, self.memory_config.enhanced_db_path)
        }
    
    def update_openai_config(self, **kwargs):
        """更新OpenAI配置"""
        for key, value in kwargs.items():
            if hasattr(self.openai_config, key):
                setattr(self.openai_config, key, value)
        self.save_config()
    
    def update_agent_config(self, **kwargs):
        """更新Agent配置"""
        for key, value in kwargs.items():
            if hasattr(self.agent_config, key):
                setattr(self.agent_config, key, value)
        self.save_config()
    
    def update_evolution_config(self, **kwargs):
        """更新进化配置"""
        for key, value in kwargs.items():
            if hasattr(self.evolution_config, key):
                setattr(self.evolution_config, key, value)
        self.save_config()
    
    def validate_config(self) -> Dict[str, bool]:
        """验证配置完整性"""
        validation_results = {}
        
        # 验证OpenAI配置
        validation_results["openai_api_key"] = bool(self.openai_config.api_key and self.openai_config.api_key != "***HIDDEN***")
        validation_results["openai_model"] = bool(self.openai_config.model)
        validation_results["openai_base_url"] = bool(self.openai_config.base_url)
        
        # 验证系统目录
        validation_results["data_directory"] = os.path.exists(self.system_config.data_directory)
        validation_results["log_directory"] = os.path.exists(self.system_config.log_directory)
        
        # 验证进化参数
        validation_results["evolution_params"] = (
            0 < self.evolution_config.mutation_rate < 1 and
            0 < self.evolution_config.crossover_rate < 1 and
            0 < self.evolution_config.elite_ratio < 1
        )
        
        return validation_results
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {
            "openai": asdict(self.openai_config),
            "agent": asdict(self.agent_config),
            "evolution": asdict(self.evolution_config),
            "memory": asdict(self.memory_config),
            "system": asdict(self.system_config)
        }
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        self.openai_config = OpenAIConfig()
        self.agent_config = AgentConfig()
        self.evolution_config = EvolutionConfig()
        self.memory_config = MemoryConfig()
        self.system_config = SystemConfig()
        
        # 重新从环境变量加载敏感配置
        self._load_from_env()
        
        self.save_config()
        logger.info("🔄 配置已重置为默认值")
    
    def export_config(self, export_path: str):
        """导出配置"""
        config_data = self.get_all_configs()
        config_data["meta"] = {
            "export_timestamp": self._get_timestamp(),
            "version": "3.0.0",
            "description": "导出的统一系统配置"
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"📤 配置已导出到: {export_path}")
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
    
    def import_config(self, import_path: str):
        """导入配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            if "openai" in config_data:
                self._update_config(self.openai_config, config_data["openai"])
            
            if "agent" in config_data:
                self._update_config(self.agent_config, config_data["agent"])
            
            if "evolution" in config_data:
                self._update_config(self.evolution_config, config_data["evolution"])
            
            if "memory" in config_data:
                self._update_config(self.memory_config, config_data["memory"])
            
            if "system" in config_data:
                self._update_config(self.system_config, config_data["system"])
            
            self.save_config()
            logger.info(f"📥 配置已从 {import_path} 导入")
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

# 全局配置管理器实例
integrated_config = IntegratedConfigManager()