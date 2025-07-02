"""
自我进化Agent安装脚本
"""
import subprocess
import sys
import os

def install_dependencies():
    """安装依赖包"""
    print("📦 正在安装依赖包...")
    
    try:
        # 尝试安装基础依赖
        basic_deps = [
            "openai>=1.3.0",
            "requests>=2.31.0", 
            "python-dotenv>=1.0.0"
        ]
        
        for dep in basic_deps:
            print(f"   安装 {dep}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        # 尝试安装可选依赖
        optional_deps = [
            "pydantic>=2.5.0",
            "colorama>=0.4.6",
            "rich>=13.7.0",
            "aiohttp>=3.9.0"
        ]
        
        for dep in optional_deps:
            try:
                print(f"   安装 {dep}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            except subprocess.CalledProcessError:
                print(f"   ⚠️ 可选依赖 {dep} 安装失败，跳过")
        
        print("✅ 依赖安装完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def create_env_file():
    """创建环境变量文件"""
    if not os.path.exists(".env"):
        print("📝 创建环境变量文件...")
        with open(".env", "w") as f:
            f.write("""# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Agent 配置
AGENT_NAME=SelfEvolvingAgent
AGENT_VERSION=1.0.0
""")
        print("✅ .env 文件已创建，请编辑并设置你的API密钥")
    else:
        print("ℹ️ .env 文件已存在，跳过创建")

def test_installation():
    """测试安装"""
    print("🧪 测试安装...")
    
    try:
        # 测试导入核心模块
        print("   测试模块导入...")
        
        # 测试基础模块
        import json
        import sqlite3
        import time
        print("   ✓ 基础模块正常")
        
        # 测试配置模块
        from config import config
        print("   ✓ 配置模块正常")
        
        # 测试记忆模块
        from memory import MemoryManager, Memory
        print("   ✓ 记忆模块正常")
        
        # 测试进化模块
        from evolution import EvolutionEngine
        print("   ✓ 进化模块正常")
        
        print("✅ 所有核心模块测试通过!")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主安装函数"""
    print("🚀 自我进化Agent安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print(f"✅ Python版本: {sys.version}")
    
    # 安装依赖
    if not install_dependencies():
        print("❌ 依赖安装失败，请手动安装")
        sys.exit(1)
    
    # 创建配置文件
    create_env_file()
    
    # 测试安装
    if not test_installation():
        print("❌ 安装测试失败")
        sys.exit(1)
    
    print("\n🎉 安装完成!")
    print("\n📝 下一步:")
    print("1. 编辑 .env 文件，设置你的 OPENAI_API_KEY")
    print("2. 运行测试: python simple_demo.py")
    print("3. 启动交互模式: python main.py")
    print("\n💡 如果遇到问题，请查看 README.md")

if __name__ == "__main__":
    main()