#!/usr/bin/env python3
"""
智能超时系统测试脚本
验证不同规模设计的超时计算
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入OpenROAD接口
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_timeout_calculation():
    """测试超时计算功能"""
    print("=== 智能超时系统测试 ===")
    
    # 测试不同规模的设计
    test_designs = [
        "mgc_des_perf_b",      # 小型设计
        "mgc_fft_1",           # 中型设计
        "mgc_superblue11_a",   # 大型设计
    ]
    
    for design_name in test_designs:
        design_path = f"data/designs/ispd_2015_contest_benchmark/{design_name}"
        
        if not os.path.exists(design_path):
            print(f"⚠️ 设计路径不存在: {design_path}")
            continue
            
        print(f"\n--- 测试设计: {design_name} ---")
        
        try:
            # 创建接口实例
            interface = RealOpenROADInterface(design_path)
            
            # 提取设计统计信息
            design_stats = interface._extract_design_stats()
            print(f"设计统计: {design_stats}")
            
            # 计算超时时间
            timeout = interface._calculate_timeout(design_stats)
            print(f"计算超时: {timeout}秒 ({timeout/60:.1f}分钟)")
            
            # 运行快速测试（只测试超时计算，不实际运行OpenROAD）
            print(f"✅ 超时计算成功")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")

def test_single_design_with_smart_timeout():
    """使用智能超时测试单个设计"""
    print("\n=== 智能超时实际测试 ===")
    
    # 使用一个较小的设计进行测试
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_b"
    
    print(f"测试设计: {design_path}")
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface(design_path)
        
        # 运行布局优化（使用智能超时）
        print("开始布局优化（使用智能超时）...")
        result = interface.run_placement(
            density_target=0.7,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        print(f"执行结果:")
        print(f"  成功: {result['success']}")
        print(f"  执行时间: {result['execution_time']:.2f}秒")
        print(f"  使用超时: {result['timeout_used']}秒")
        print(f"  设计统计: {result['design_stats']}")
        
        if result['success']:
            print("✅ 智能超时测试成功！")
        else:
            print("❌ 智能超时测试失败")
            if result['stderr']:
                print(f"错误信息: {result['stderr'][-1]}")
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")

def analyze_timeout_performance():
    """分析超时性能"""
    print("\n=== 超时性能分析 ===")
    
    # 读取之前的训练结果
    summary_file = "results/ispd_training/training_summary.json"
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("之前的训练结果分析:")
        print(f"  总设计数: {summary['training_info']['total_designs']}")
        print(f"  成功设计: {summary['training_info']['successful_designs']}")
        print(f"  失败设计: {summary['training_info']['failed_designs']}")
        print(f"  成功率: {summary['training_info']['successful_designs']/summary['training_info']['total_designs']*100:.1f}%")
        print(f"  平均时间: {summary['training_info']['average_time']:.2f}秒")
        
        # 分析失败原因
        failed_designs = summary.get('failed_designs', [])
        timeout_failures = 0
        
        for design_name in failed_designs:
            log_file = f"results/ispd_training/{design_name}_log.txt"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "超时" in content or "timeout" in content.lower():
                        timeout_failures += 1
        
        print(f"  超时失败: {timeout_failures}/{len(failed_designs)}")
        
        if timeout_failures > 0:
            print("\n建议:")
            print("  - 对于大型设计，增加超时时间")
            print("  - 对于小型设计，减少超时时间")
            print("  - 使用智能超时系统优化性能")
    else:
        print("未找到训练结果文件")

def main():
    """主函数"""
    print("开始智能超时系统测试...")
    
    # 测试超时计算
    test_timeout_calculation()
    
    # 测试实际运行
    test_single_design_with_smart_timeout()
    
    # 分析性能
    analyze_timeout_performance()
    
    print("\n🎉 智能超时系统测试完成！")

if __name__ == "__main__":
    main() 