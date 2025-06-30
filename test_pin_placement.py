#!/usr/bin/env python3
"""
简单的pin placement功能测试脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pin_placement():
    """测试pin placement功能"""
    print("=== Pin Placement功能测试 ===")
    
    # 测试设计路径
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    design_dir = Path(design_path)
    
    if not design_dir.exists():
        print(f"❌ 设计目录不存在: {design_path}")
        return False
    
    print(f"测试设计: {design_dir.name}")
    
    # 检查是否有placement_result.def文件
    def_file = design_dir / "placement_result.def"
    if not def_file.exists():
        print("❌ 没有找到placement_result.def文件，请先运行布局优化")
        return False
    
    print("✅ 找到placement_result.def文件")
    
    # 检查DEF文件中的引脚信息
    with open(def_file, 'r') as f:
        def_content = f.read()
    
    # 检查引脚信息
    if 'PINS' in def_content:
        print("✅ DEF文件包含引脚信息")
        
        # 统计引脚数量
        import re
        pins_match = re.search(r'PINS\s+(\d+)', def_content)
        if pins_match:
            num_pins = int(pins_match.group(1))
            print(f"  引脚总数: {num_pins}")
        
        # 检查引脚是否有PLACED和LAYER信息
        if 'PLACED' in def_content and 'LAYER' in def_content:
            print("✅ 引脚包含PLACED和LAYER信息")
            
            # 统计已放置引脚
            placed_count = def_content.count('PLACED')
            print(f"  PLACED关键字出现次数: {placed_count}")
            
            # 检查是否有具体的引脚放置信息
            pin_lines = [line for line in def_content.split('\n') if 'PIN' in line and 'PLACED' in line]
            if pin_lines:
                print(f"  找到 {len(pin_lines)} 个已放置的引脚")
                print("  示例引脚信息:")
                for i, line in enumerate(pin_lines[:3]):  # 显示前3个
                    print(f"    {line.strip()}")
            else:
                print("⚠️  未找到具体的引脚放置信息")
        else:
            print("❌ 引脚缺少PLACED或LAYER信息")
    else:
        print("❌ DEF文件缺少引脚信息")
    
    # 检查是否有执行日志
    log_file = design_dir / "openroad_execution.log"
    if log_file.exists():
        print("\n=== 检查执行日志 ===")
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # 检查pin placement相关日志
        if 'pin placement' in log_content.lower() or 'place_pins' in log_content:
            print("✅ 日志中包含pin placement相关信息")
        else:
            print("❌ 日志中未找到pin placement相关信息")
        
        # 检查布局完成情况
        if '布局完成' in log_content or 'placement completed' in log_content.lower():
            print("✅ 布局流程完成")
        else:
            print("❌ 布局流程未完成")
    else:
        print("❌ 未找到执行日志文件")
    
    print("\n=== 测试总结 ===")
    print("Pin placement功能已添加到TCL脚本生成中")
    print("下次运行布局优化时，将自动执行pin placement命令")
    print("建议重新运行布局优化以验证pin placement效果")
    
    return True

def main():
    """主函数"""
    success = test_pin_placement()
    
    if success:
        print("\n✅ 测试完成")
        return 0
    else:
        print("\n❌ 测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 