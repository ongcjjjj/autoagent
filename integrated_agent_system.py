"""
完整整合的自主进化Agent系统
整合所有模块和功能的统一入口点
"""

import asyncio
import time
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional, AsyncGenerator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入所有必要的模块
try:
    from config import config
    from memory import Memory, MemoryManager
    from openai_client import openai_client
    from unified_evolution_system import UnifiedEvolutionSystem, EnhancedMemory
    from unified_agent import UnifiedSelfEvolvingAgent
    logger.info("✅ 成功导入所有核心模块")
except ImportError as e:
    logger.error(f"❌ 导入模块失败: {e}")
    sys.exit(1)

class IntegratedAgentSystem:
    """完整整合的自主进化Agent系统"""
    
    def __init__(self):
        """初始化完整系统"""
        logger.info("🚀 初始化完整整合的自主进化Agent系统...")
        
        # 初始化核心组件
        self.unified_agent = UnifiedSelfEvolvingAgent()
        self.evolution_system = self.unified_agent.evolution_system
        self.memory_manager = self.unified_agent.memory_manager
        
        # 系统状态
        self.is_running = False
        self.startup_time = time.time()
        
        # 统计信息
        self.interaction_count = 0
        self.evolution_count = 0
        self.error_count = 0
        
        logger.info(f"✅ 系统初始化完成 - {self.unified_agent.name} v{self.unified_agent.version}")
    
    async def start_system(self):
        """启动完整系统"""
        logger.info("🔥 启动完整自主进化Agent系统...")
        
        # 系统健康检查
        health_check = await self.system_health_check()
        if not all(health_check.values()):
            logger.error("❌ 系统健康检查失败")
            for component, status in health_check.items():
                if not status:
                    logger.error(f"  - {component}: 失败")
            return False
        
        self.is_running = True
        logger.info("✅ 系统启动成功，所有组件正常运行")
        return True
    
    async def process_user_input(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """处理用户输入的统一接口"""
        if not self.is_running:
            return {"error": "系统未启动", "content": "请先启动系统"}
        
        start_time = time.time()
        self.interaction_count += 1
        
        try:
            # 使用统一代理处理消息
            response = await self.unified_agent.process_message(
                user_message=user_input,
                context=context,
                stream=False
            )
            
            # 更新统计信息
            if response.get('error'):
                self.error_count += 1
            
            # 添加系统信息
            response['system_info'] = {
                'interaction_count': self.interaction_count,
                'response_time': time.time() - start_time,
                'system_uptime': time.time() - self.startup_time,
                'evolution_count': len(self.evolution_system.evolution_history)
            }
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"处理用户输入时发生错误: {e}")
            return {
                "error": True,
                "content": f"处理过程中发生错误: {str(e)}",
                "error_details": str(e)
            }
    
    async def stream_response(self, user_input: str, context: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """流式响应接口"""
        if not self.is_running:
            yield "系统未启动，请先启动系统"
            return
        
        self.interaction_count += 1
        
        try:
            response_generator = await self.unified_agent.process_message(
                user_message=user_input,
                context=context,
                stream=True
            )
            
            async for chunk in response_generator:
                yield chunk
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"流式响应时发生错误: {e}")
            yield f"错误: {str(e)}"
    
    def trigger_evolution(self) -> Dict[str, Any]:
        """手动触发进化"""
        try:
            evolution_result = self.unified_agent.trigger_manual_evolution()
            self.evolution_count += 1
            logger.info(f"🧬 手动进化完成 - 策略: {evolution_result.get('strategy', 'unknown')}")
            return {
                "success": True,
                "evolution_result": evolution_result,
                "total_evolutions": self.evolution_count
            }
        except Exception as e:
            logger.error(f"进化过程中发生错误: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取全面的系统状态"""
        return {
            "system": {
                "is_running": self.is_running,
                "uptime": time.time() - self.startup_time,
                "interaction_count": self.interaction_count,
                "evolution_count": self.evolution_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.interaction_count, 1)
            },
            "agent_status": self.unified_agent.get_enhanced_status(),
            "evolution_info": self.unified_agent._get_evolution_info(),
            "memory_stats": self._get_memory_statistics()
        }
    
    def _get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            basic_stats = self.memory_manager.get_memory_stats()
            enhanced_stats = self.evolution_system.get_system_status()
            
            return {
                "basic_memory": basic_stats,
                "enhanced_memory": {
                    "database_path": enhanced_stats.get("database_path"),
                    "performance_window_size": enhanced_stats.get("performance_window_size"),
                    "evolution_count": enhanced_stats.get("evolution_count")
                }
            }
        except Exception as e:
            logger.error(f"获取记忆统计失败: {e}")
            return {"error": str(e)}
    
    async def system_health_check(self) -> Dict[str, bool]:
        """系统健康检查"""
        health_status = {}
        
        # 检查统一代理
        try:
            test_results = await self.unified_agent.test_all_systems()
            health_status.update(test_results)
        except Exception as e:
            logger.error(f"代理系统检查失败: {e}")
            health_status["unified_agent"] = False
        
        # 检查进化系统
        try:
            evo_status = self.evolution_system.get_system_status()
            health_status["evolution_system"] = True
        except Exception as e:
            logger.error(f"进化系统检查失败: {e}")
            health_status["evolution_system"] = False
        
        # 检查配置
        try:
            config_info = config.get_openai_client_kwargs()
            health_status["configuration"] = bool(config_info.get("api_key"))
        except Exception as e:
            logger.error(f"配置检查失败: {e}")
            health_status["configuration"] = False
        
        return health_status
    
    def export_complete_system_data(self, export_path: str = "complete_system_export"):
        """导出完整系统数据"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_dir = f"{export_path}_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            # 导出统一代理数据
            agent_export_path = os.path.join(export_dir, "unified_agent_data.json")
            self.unified_agent.export_enhanced_data(agent_export_path)
            
            # 导出基础记忆
            memory_export_path = os.path.join(export_dir, "basic_memories.json")
            self.memory_manager.export_memories(memory_export_path)
            
            # 导出系统状态
            status_export_path = os.path.join(export_dir, "system_status.json")
            with open(status_export_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_comprehensive_status(), f, indent=2, ensure_ascii=False)
            
            # 导出配置信息
            config_export_path = os.path.join(export_dir, "system_config.json")
            config_data = {
                "openai_config": config.openai_config.model_dump(),
                "agent_config": config.agent_config.model_dump(),
                "export_timestamp": time.time()
            }
            with open(config_export_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 完整系统数据已导出到: {export_dir}")
            return export_dir
            
        except Exception as e:
            logger.error(f"导出系统数据失败: {e}")
            return None
    
    def cleanup_system(self):
        """清理系统资源"""
        try:
            # 清理统一代理
            self.unified_agent.cleanup_enhanced()
            
            # 保存进化数据
            self.evolution_system.save_evolution_data()
            
            self.is_running = False
            logger.info("✅ 系统清理完成")
            
        except Exception as e:
            logger.error(f"系统清理时发生错误: {e}")
    
    async def run_comprehensive_demo(self):
        """运行全面的系统演示"""
        logger.info("🎬 开始完整系统演示...")
        
        # 启动系统
        if not await self.start_system():
            return False
        
        demo_results = []
        
        # 演示1: 基础对话
        logger.info("演示1: 基础对话功能")
        response1 = await self.process_user_input("你好，请介绍一下你的能力")
        demo_results.append({"demo": "basic_conversation", "success": not response1.get("error", False)})
        
        # 演示2: 复杂问题处理
        logger.info("演示2: 复杂问题处理")
        response2 = await self.process_user_input("请解释机器学习中的梯度下降算法原理并给出Python实现")
        demo_results.append({"demo": "complex_question", "success": not response2.get("error", False)})
        
        # 演示3: 手动触发进化
        logger.info("演示3: 手动触发进化")
        evolution_result = self.trigger_evolution()
        demo_results.append({"demo": "manual_evolution", "success": evolution_result.get("success", False)})
        
        # 演示4: 系统状态检查
        logger.info("演示4: 系统状态检查")
        status = self.get_comprehensive_status()
        demo_results.append({"demo": "system_status", "success": status.get("system", {}).get("is_running", False)})
        
        # 演示5: 数据导出
        logger.info("演示5: 数据导出")
        export_path = self.export_complete_system_data("demo_export")
        demo_results.append({"demo": "data_export", "success": export_path is not None})
        
        # 汇总结果
        success_count = sum(1 for result in demo_results if result["success"])
        total_demos = len(demo_results)
        
        logger.info(f"🎯 演示完成: {success_count}/{total_demos} 成功")
        
        # 保存演示结果
        demo_summary = {
            "timestamp": time.time(),
            "total_demos": total_demos,
            "successful_demos": success_count,
            "success_rate": success_count / total_demos,
            "detailed_results": demo_results,
            "system_status": self.get_comprehensive_status()
        }
        
        with open("integrated_system_demo_results.json", "w", encoding="utf-8") as f:
            json.dump(demo_summary, f, indent=2, ensure_ascii=False)
        
        return success_count == total_demos

# 全局系统实例
integrated_system = IntegratedAgentSystem()

async def main():
    """主函数 - 演示完整系统"""
    try:
        # 运行全面演示
        success = await integrated_system.run_comprehensive_demo()
        
        if success:
            logger.info("🎉 完整系统演示成功！")
        else:
            logger.error("❌ 部分演示失败")
        
        # 清理系统
        integrated_system.cleanup_system()
        
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())