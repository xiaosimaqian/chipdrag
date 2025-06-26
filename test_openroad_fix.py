#!/usr/bin/env python3
"""
测试OpenROAD接口修复
验证数据库重置和错误处理是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
import json

def test_openroad_interface():
    """测试OpenROAD接口"""
    print("=== 测试OpenROAD接口修复 ===")
    
    # 初始化接口
    try:
        interface = RealOpenROADInterface()
        print("✓ 接口初始化成功")
    except Exception as e:
        print(f"✗ 接口初始化失败: {e}")
        return
    
    # 测试第一次运行
    print("\n--- 第一次运行 ---")
    result1 = interface.run_placement(density_target=0.7)
    
    print(f"成功: {result1['success']}")
    print(f"返回码: {result1['return_code']}")
    print(f"执行时间: {result1['execution_time']:.2f}秒")
    
    if result1['stderr']:
        print("错误输出:")
        for line in result1['stderr'][:5]:  # 只显示前5行
            print(f"  {line}")
    
    # 测试第二次运行（验证数据库重置）
    print("\n--- 第二次运行 ---")
    result2 = interface.run_placement(density_target=0.8)
    
    print(f"成功: {result2['success']}")
    print(f"返回码: {result2['return_code']}")
    print(f"执行时间: {result2['execution_time']:.2f}秒")
    
    if result2['stderr']:
        print("错误输出:")
        for line in result2['stderr'][:5]:  # 只显示前5行
            print(f"  {line}")
    
    # 检查是否有"Chip already exists"错误
    chip_exists_error = False
    for result in [result1, result2]:
        for line in result['stderr']:
            if 'Chip already exists' in line:
                chip_exists_error = True
                break
    
    if chip_exists_error:
        print("\n✗ 仍然存在'Chip already exists'错误")
    else:
        print("\n✓ 没有'Chip already exists'错误")
    
    # 保存详细结果
    with open('openroad_test_results.json', 'w') as f:
        json.dump({
            'result1': result1,
            'result2': result2,
            'chip_exists_error': chip_exists_error
        }, f, indent=2)
    
    print(f"\n详细结果已保存到: openroad_test_results.json")

if __name__ == "__main__":
    test_openroad_interface() 