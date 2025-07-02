#!/usr/bin/env python3
"""
部署脚本 - 自主进化Agent系统
自动化部署和环境配置
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    
    print(f"✅ Python版本检查通过: {sys.version.split()[0]}")
    return True


def check_dependencies():
    """检查依赖"""
    print("\n📦 检查依赖...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        missing_deps = []
        for req in requirements:
            if req.strip():
                package = req.split('==')[0].split('>=')[0].split('<=')[0]
                try:
                    __import__(package)
                    print(f"✅ {package}")
                except ImportError:
                    missing_deps.append(req)
                    print(f"❌ {package} (未安装)")
        
        if missing_deps:
            print(f"\n⚠️  发现 {len(missing_deps)} 个缺失的依赖")
            return False, missing_deps
        else:
            print("✅ 所有依赖已满足")
            return True, []
            
    except FileNotFoundError:
        print("❌ requirements.txt 文件未找到")
        return False, []


def install_dependencies(missing_deps):
    """安装缺失的依赖"""
    print("\n📥 安装缺失的依赖...")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install'] + missing_deps
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖安装成功")
            return True
        else:
            print(f"❌ 依赖安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 安装过程中出错: {e}")
        return False


def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        'data/system_states',
        'data/logs',
        'data/backups',
        'config',
        'scripts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")


def create_config_files():
    """创建配置文件"""
    print("\n⚙️  创建配置文件...")
    
    # 系统配置
    system_config = {
        "system": {
            "name": "autonomous_evolutionary_agents",
            "version": "1.0.0",
            "log_level": "INFO"
        },
        "agents": {
            "max_memory_size": 100,
            "performance_history_size": 50,
            "success_pattern_limit": 20,
            "optimization_threshold": 0.1
        },
        "communication": {
            "timeout": 30.0,
            "max_message_size": 1048576,
            "retry_attempts": 3
        },
        "evaluation": {
            "weights": {
                "trainability": 0.15,
                "generalization": 0.15,
                "expressiveness": 0.10,
                "creativity_score": 0.15,
                "adaptation_rate": 0.10,
                "collaboration_efficiency": 0.15,
                "error_recovery_rate": 0.10,
                "knowledge_retention": 0.05,
                "innovation_index": 0.05
            }
        }
    }
    
    config_path = Path('config/system_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(system_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {config_path}")
    
    # 日志配置
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "data/logs/system.log",
                "mode": "a"
            }
        },
        "loggers": {
            "autonomous_evolutionary_agent_system": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"]
        }
    }
    
    log_config_path = Path('config/logging_config.json')
    with open(log_config_path, 'w', encoding='utf-8') as f:
        json.dump(log_config, f, indent=2)
    
    print(f"✅ {log_config_path}")


def run_tests():
    """运行测试套件"""
    print("\n🧪 运行测试套件...")
    
    test_files = [
        'tests/test_simplified.py',
        'tests/test_communication.py'
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n运行 {test_file}...")
            try:
                result = subprocess.run(
                    [sys.executable, test_file], 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print(f"✅ {test_file} 通过")
                else:
                    print(f"❌ {test_file} 失败")
                    print(result.stderr)
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} 超时")
                all_passed = False
            except Exception as e:
                print(f"❌ {test_file} 执行错误: {e}")
                all_passed = False
        else:
            print(f"⚠️  {test_file} 不存在，跳过")
    
    return all_passed


def verify_installation():
    """验证安装"""
    print("\n🔍 验证安装...")
    
    try:
        # 测试导入主模块
        sys.path.insert(0, '.')
        from autonomous_evolutionary_agent_system import AutonomousEvolutionarySystem
        
        # 创建系统实例
        system = AutonomousEvolutionarySystem()
        
        # 创建标准团队
        team = system.create_standard_team()
        
        print(f"✅ 系统创建成功，包含 {len(team)} 个Agent")
        
        # 检查Agent类型
        expected_roles = ['researcher', 'executor', 'critic', 'coordinator', 'architect']
        for role in expected_roles:
            if role in team:
                print(f"✅ {role} Agent 创建成功")
            else:
                print(f"❌ {role} Agent 创建失败")
                return False
        
        print("✅ 安装验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 安装验证失败: {e}")
        return False


def create_startup_script():
    """创建启动脚本"""
    print("\n📝 创建启动脚本...")
    
    startup_script = '''#!/usr/bin/env python3
"""
启动脚本 - 自主进化Agent系统
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from autonomous_evolutionary_agent_system import AutonomousEvolutionarySystem


async def main():
    """主函数"""
    print("🚀 启动自主进化Agent系统...")
    
    # 创建系统
    system = AutonomousEvolutionarySystem()
    
    # 创建标准团队
    team = system.create_standard_team()
    print(f"✅ 创建了 {len(team)} 个Agent")
    
    # 运行示例任务
    result = await system.run_collaborative_task(
        goal="系统启动验证任务",
        max_cycles=2
    )
    
    print(f"✅ 任务完成，性能得分: {result['final_metrics'].composite_score:.3f}")
    print("🎉 系统运行正常！")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    startup_path = Path('start.py')
    with open(startup_path, 'w', encoding='utf-8') as f:
        f.write(startup_script)
    
    # 设置执行权限
    startup_path.chmod(0o755)
    print(f"✅ {startup_path}")


def main():
    """主部署函数"""
    parser = argparse.ArgumentParser(description='自主进化Agent系统部署脚本')
    parser.add_argument('--skip-tests', action='store_true', help='跳过测试')
    parser.add_argument('--force-install', action='store_true', help='强制重新安装依赖')
    args = parser.parse_args()
    
    print("🚀 自主进化Agent系统部署脚本")
    print("=" * 50)
    
    # 1. 检查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 2. 检查依赖
    deps_ok, missing_deps = check_dependencies()
    
    if not deps_ok or args.force_install:
        if args.force_install or missing_deps:
            if not install_dependencies(missing_deps if missing_deps else ['numpy']):
                print("❌ 依赖安装失败，部署中止")
                sys.exit(1)
    
    # 3. 创建目录结构
    create_directories()
    
    # 4. 创建配置文件
    create_config_files()
    
    # 5. 创建启动脚本
    create_startup_script()
    
    # 6. 运行测试
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  部分测试失败，但部署继续")
    
    # 7. 验证安装
    if verify_installation():
        print("\n🎉 部署成功完成！")
        print("\n📋 后续步骤:")
        print("1. 运行 'python start.py' 启动系统")
        print("2. 查看 'examples/' 目录了解使用方法")
        print("3. 阅读 'docs/' 目录中的文档")
        print("4. 检查 'data/logs/' 目录中的日志")
    else:
        print("❌ 部署失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()