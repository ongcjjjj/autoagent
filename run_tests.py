#!/usr/bin/env python3
"""
测试运行脚本 - 自主进化Agent系统
运行所有测试并生成报告
"""

import os
import sys
import unittest
import time
from pathlib import Path
from io import StringIO


def discover_and_run_tests():
    """发现并运行所有测试"""
    print("🧪 自主进化Agent系统 - 测试套件")
    print("=" * 60)
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # 发现测试
    test_dir = project_root / 'tests'
    if not test_dir.exists():
        print("❌ 测试目录不存在")
        return False
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 手动添加测试文件
    test_files = [
        'test_simplified.py',
        'test_communication.py'
    ]
    
    total_tests = 0
    loaded_files = 0
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            try:
                # 导入测试模块
                module_name = f"tests.{test_file[:-3]}"
                module = __import__(module_name, fromlist=[test_file[:-3]])
                
                # 加载测试
                file_suite = loader.loadTestsFromModule(module)
                suite.addTest(file_suite)
                
                # 计算测试数量
                test_count = file_suite.countTestCases()
                total_tests += test_count
                loaded_files += 1
                
                print(f"✅ 加载 {test_file}: {test_count} 个测试")
                
            except Exception as e:
                print(f"❌ 加载 {test_file} 失败: {e}")
        else:
            print(f"⚠️  {test_file} 不存在，跳过")
    
    print(f"\n📊 测试统计:")
    print(f"   加载文件: {loaded_files}")
    print(f"   总测试数: {total_tests}")
    
    if total_tests == 0:
        print("❌ 没有找到可运行的测试")
        return False
    
    # 运行测试
    print(f"\n🚀 开始运行测试...")
    start_time = time.time()
    
    # 创建测试运行器
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    # 运行测试
    result = runner.run(suite)
    
    # 计算运行时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 显示结果
    print(f"\n📋 测试结果:")
    print(f"   运行时间: {duration:.2f} 秒")
    print(f"   运行测试: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    print(f"   跳过: {len(result.skipped)}")
    
    # 显示详细结果
    if result.failures:
        print(f"\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Error:')[-1].strip()}")
    
    # 计算成功率
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📊 成功率: {success_rate:.1f}%")
    
    # 判断整体结果
    if result.wasSuccessful():
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️  部分测试失败")
        return False


def run_specific_test(test_name):
    """运行特定测试"""
    print(f"🎯 运行特定测试: {test_name}")
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    try:
        # 导入并运行特定测试
        if test_name.endswith('.py'):
            test_name = test_name[:-3]
        
        module_name = f"tests.{test_name}"
        module = __import__(module_name, fromlist=[test_name])
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return False


def run_system_verification():
    """运行系统验证测试"""
    print("🔍 系统验证测试")
    print("-" * 30)
    
    try:
        # 添加项目根目录到路径
        project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(project_root))
        
        # 导入系统
        from autonomous_evolutionary_agent_system import AutonomousEvolutionarySystem
        
        print("✅ 主模块导入成功")
        
        # 创建系统
        system = AutonomousEvolutionarySystem()
        print("✅ 系统实例创建成功")
        
        # 创建团队
        team = system.create_standard_team()
        print(f"✅ 标准团队创建成功 ({len(team)} 个Agent)")
        
        # 验证Agent类型
        expected_roles = ['researcher', 'executor', 'critic', 'coordinator', 'architect']
        for role in expected_roles:
            if role in team:
                print(f"✅ {role.capitalize()} Agent 验证通过")
            else:
                print(f"❌ {role.capitalize()} Agent 验证失败")
                return False
        
        print("🎉 系统验证完成，所有组件正常")
        return True
        
    except Exception as e:
        print(f"❌ 系统验证失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自主进化Agent系统测试运行器')
    parser.add_argument('--test', '-t', help='运行特定测试文件')
    parser.add_argument('--verify', '-v', action='store_true', help='只运行系统验证')
    parser.add_argument('--quick', '-q', action='store_true', help='快速测试模式')
    
    args = parser.parse_args()
    
    if args.verify:
        # 只运行系统验证
        success = run_system_verification()
        sys.exit(0 if success else 1)
    
    elif args.test:
        # 运行特定测试
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    else:
        # 运行完整测试套件
        if args.quick:
            print("⚡ 快速测试模式")
        
        # 先运行系统验证
        print("第一阶段: 系统验证")
        if not run_system_verification():
            print("❌ 系统验证失败，终止测试")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("第二阶段: 单元测试")
        
        # 运行完整测试
        success = discover_and_run_tests()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 所有测试完成，系统状态良好！")
            print("\n💡 提示:")
            print("- 运行 'python start.py' 启动系统")
            print("- 查看 'examples/' 目录了解使用方法")
        else:
            print("⚠️  测试完成，但发现问题")
            print("请检查失败的测试并修复相关问题")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()