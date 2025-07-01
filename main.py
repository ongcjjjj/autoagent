"""
主程序入口 - 增强版
提供智能命令行界面、实时监控、用户管理、性能分析
"""
import asyncio
import argparse
import sys
import os
import uuid
import json
import time
import threading
from typing import Optional, Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.align import Align
from rich.columns import Columns
from datetime import datetime, timedelta
from collections import defaultdict, deque

from agent import SelfEvolvingAgent
from config import config

console = Console()

class EnhancedAgentCLI:
    """增强版Agent命令行界面"""

class AgentCLI:
    """Agent命令行界面"""
    
    def __init__(self):
        self.agent: Optional[SelfEvolvingAgent] = None
        self.running = False
    
    async def start_agent(self, name: Optional[str] = None):
        """启动Agent"""
        console.print("🚀 正在启动自我进化Agent...", style="bold blue")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("初始化中...", total=None)
            
            try:
                self.agent = SelfEvolvingAgent(name=name)
                progress.update(task, description="测试API连接...")
                
                connection_ok = await self.agent.test_connection()
                if not connection_ok:
                    console.print("❌ API连接失败，请检查配置", style="bold red")
                    return False
                
                progress.update(task, description="启动完成")
                
            except Exception as e:
                console.print(f"❌ 启动失败: {e}", style="bold red")
                return False
        
        console.print("✅ Agent启动成功!", style="bold green")
        return True
    
    async def interactive_mode(self):
        """交互模式"""
        if not self.agent:
            console.print("❌ Agent未启动", style="bold red")
            return
        
        console.print(Panel.fit(
            f"🤖 欢迎使用 {self.agent.name}!\n"
            "输入 /help 查看命令帮助\n"
            "输入 /quit 退出程序",
            title="交互模式",
            style="bold blue"
        ))
        
        self.running = True
        
        while self.running:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if not user_input.strip():
                    continue
                
                if user_input.startswith('/'):
                    await self.handle_command(user_input)
                else:
                    await self.handle_message(user_input)
                    
            except KeyboardInterrupt:
                if Confirm.ask("\n确定要退出吗?"):
                    self.running = False
            except Exception as e:
                console.print(f"❌ 错误: {e}", style="bold red")
    
    async def handle_command(self, command: str):
        """处理命令"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/status":
            await self.show_status()
        elif cmd == "/config":
            await self.handle_config_command(parts[1:] if len(parts) > 1 else [])
        elif cmd == "/memory":
            await self.handle_memory_command(parts[1:] if len(parts) > 1 else [])
        elif cmd == "/evolution":
            await self.show_evolution()
        elif cmd == "/export":
            await self.export_data(parts[1] if len(parts) > 1 else "agent_export.json")
        elif cmd == "/quit" or cmd == "/exit":
            self.running = False
        else:
            console.print(f"❌ 未知命令: {cmd}", style="bold red")
    
    def show_help(self):
        """显示帮助"""
        help_table = Table(title="命令帮助")
        help_table.add_column("命令", style="cyan")
        help_table.add_column("描述", style="white")
        
        commands = [
            ("/help", "显示此帮助信息"),
            ("/status", "显示Agent状态"),
            ("/config [key] [value]", "查看或设置配置"),
            ("/memory search <query>", "搜索记忆"),
            ("/memory add <content>", "添加记忆"),
            ("/evolution", "显示进化历史"),
            ("/export [filename]", "导出数据"),
            ("/quit", "退出程序")
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        console.print(help_table)
    
    async def show_status(self):
        """显示状态"""
        status = self.agent.get_status()
        
        # Agent信息
        agent_table = Table(title="Agent信息")
        agent_table.add_column("属性", style="cyan")
        agent_table.add_column("值", style="white")
        
        agent_info = status["agent"]
        agent_table.add_row("名称", agent_info["name"])
        agent_table.add_row("版本", agent_info["version"])
        agent_table.add_row("个性风格", str(agent_info["personality"]))
        
        # 记忆统计
        memory_table = Table(title="记忆统计")
        memory_table.add_column("指标", style="cyan")
        memory_table.add_column("数值", style="white")
        
        memory_info = status["memory"]
        memory_table.add_row("总记忆数", str(memory_info["total_memories"]))
        memory_table.add_row("重要记忆数", str(memory_info["important_memories"]))
        memory_table.add_row("类型分布", str(memory_info["type_distribution"]))
        
        # 进化信息
        evolution_table = Table(title="进化状态")
        evolution_table.add_column("指标", style="cyan")
        evolution_table.add_column("值", style="white")
        
        evolution_info = status["evolution"]
        if "message" not in evolution_info:
            evolution_table.add_row("进化次数", str(evolution_info["total_evolutions"]))
            evolution_table.add_row("当前版本", evolution_info["latest_version"])
            evolution_table.add_row("性能趋势", evolution_info["performance_trend"])
        else:
            evolution_table.add_row("状态", evolution_info["message"])
        
        console.print(agent_table)
        console.print(memory_table)
        console.print(evolution_table)
    
    async def handle_config_command(self, args: list):
        """处理配置命令"""
        if not args:
            # 显示当前配置
            config_table = Table(title="当前配置")
            config_table.add_column("配置项", style="cyan")
            config_table.add_column("值", style="white")
            
            config_table.add_row("API Key", config.openai_config.api_key[:10] + "..." if config.openai_config.api_key else "未设置")
            config_table.add_row("Base URL", config.openai_config.base_url)
            config_table.add_row("Model", config.openai_config.model)
            config_table.add_row("Max Tokens", str(config.openai_config.max_tokens))
            config_table.add_row("Temperature", str(config.openai_config.temperature))
            
            console.print(config_table)
        
        elif len(args) == 2:
            key, value = args
            try:
                if key == "api_key":
                    self.agent.update_config(api_key=value)
                elif key == "base_url":
                    self.agent.update_config(base_url=value)
                elif key == "model":
                    self.agent.update_config(model=value)
                elif key == "max_tokens":
                    self.agent.update_config(max_tokens=int(value))
                elif key == "temperature":
                    self.agent.update_config(temperature=float(value))
                else:
                    console.print(f"❌ 未知配置项: {key}", style="bold red")
                    return
                
                console.print(f"✅ 配置已更新: {key} = {value}", style="bold green")
            except Exception as e:
                console.print(f"❌ 配置更新失败: {e}", style="bold red")
        else:
            console.print("❌ 用法: /config [key] [value]", style="bold red")
    
    async def handle_memory_command(self, args: list):
        """处理记忆命令"""
        if not args:
            console.print("❌ 用法: /memory search <query> 或 /memory add <content>", style="bold red")
            return
        
        if args[0] == "search" and len(args) > 1:
            query = " ".join(args[1:])
            memories = self.agent.search_memory(query)
            
            if not memories:
                console.print("未找到相关记忆", style="yellow")
                return
            
            memory_table = Table(title=f"搜索结果: '{query}'")
            memory_table.add_column("ID", style="cyan")
            memory_table.add_column("类型", style="green")
            memory_table.add_column("重要性", style="yellow")
            memory_table.add_column("内容", style="white")
            
            for memory in memories[:10]:  # 限制显示数量
                content = memory["content"][:50] + "..." if len(memory["content"]) > 50 else memory["content"]
                memory_table.add_row(
                    str(memory["id"]),
                    memory["type"],
                    f"{memory['importance']:.2f}",
                    content
                )
            
            console.print(memory_table)
        
        elif args[0] == "add" and len(args) > 1:
            content = " ".join(args[1:])
            memory_id = self.agent.add_manual_memory(content)
            console.print(f"✅ 记忆已添加，ID: {memory_id}", style="bold green")
        else:
            console.print("❌ 用法: /memory search <query> 或 /memory add <content>", style="bold red")
    
    async def show_evolution(self):
        """显示进化历史"""
        history = self.agent.get_evolution_history()
        
        if not history:
            console.print("尚无进化历史", style="yellow")
            return
        
        evolution_table = Table(title="进化历史")
        evolution_table.add_column("版本", style="cyan")
        evolution_table.add_column("成功率", style="green")
        evolution_table.add_column("响应质量", style="yellow")
        evolution_table.add_column("改进领域", style="white")
        
        for record in history[-10:]:  # 显示最近10次进化
            evolution_table.add_row(
                record["version"],
                f"{record['metrics']['success_rate']:.2f}",
                f"{record['metrics']['response_quality']:.2f}",
                ", ".join(record["improvements"])
            )
        
        console.print(evolution_table)
    
    async def export_data(self, filename: str):
        """导出数据"""
        try:
            self.agent.export_data(filename)
            console.print(f"✅ 数据已导出到 {filename}", style="bold green")
        except Exception as e:
            console.print(f"❌ 导出失败: {e}", style="bold red")
    
    async def handle_message(self, message: str):
        """处理用户消息"""
        console.print(f"\n[bold green]{self.agent.name}[/bold green]: ", end="")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("思考中..."),
                console=console,
            ) as progress:
                task = progress.add_task("", total=None)
                response = await self.agent.process_message(message)
            
            if response.get("error"):
                console.print(f"❌ {response['content']}", style="bold red")
            else:
                console.print(response["content"])
        
        except Exception as e:
            console.print(f"❌ 处理消息时出错: {e}", style="bold red")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自我进化Agent")
    parser.add_argument("--name", help="Agent名称")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--base-url", help="OpenAI API Base URL")
    parser.add_argument("--model", help="使用的模型")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.api_key:
        config.update_openai_config(api_key=args.api_key)
    if args.base_url:
        config.update_openai_config(base_url=args.base_url)
    if args.model:
        config.update_openai_config(model=args.model)
    
    # 启动CLI
    cli = AgentCLI()
    
    if await cli.start_agent(name=args.name):
        await cli.interactive_mode()
    
    console.print("👋 再见!", style="bold blue")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 程序已退出", style="bold blue")
    except Exception as e:
        console.print(f"\n❌ 程序错误: {e}", style="bold red")
        sys.exit(1)