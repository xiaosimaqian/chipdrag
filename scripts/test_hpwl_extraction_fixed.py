#!/usr/bin/env python3
"""
测试修复后的HPWL提取函数
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment import UnifiedPaperExperiment

def test_hpwl_extraction():
    """测试HPWL提取函数"""
    print("开始测试HPWL提取函数...")
    
    # 初始化实验类
    experiment = UnifiedPaperExperiment()
    
    # 测试DEF文件路径
    def_file = Path("dataset/ispd_2015_contest_benchmark/mgc_des_perf_b/placed.def")
    
    if not def_file.exists():
        print(f"错误：DEF文件不存在: {def_file}")
        return
    
    print(f"测试DEF文件: {def_file}")
    
    # 测试ISPD2005风格的HPWL提取
    print("\n=== 测试ISPD2005风格HPWL提取 ===")
    hpwl = experiment._extract_hpwl_from_def_ispd2005_style(def_file)
    
    if hpwl is not None:
        print(f"✅ HPWL提取成功: {hpwl:.0f}")
        print(f"   实际微米值: {hpwl/1000.0:.2f}")
    else:
        print("❌ HPWL提取失败")
    
    # 测试原始HPWL提取方法作为对比
    print("\n=== 测试原始HPWL提取方法 ===")
    hpwl_original = experiment._extract_hpwl_from_def(def_file)
    
    if hpwl_original is not None:
        print(f"✅ 原始方法HPWL: {hpwl_original:.0f}")
        print(f"   实际微米值: {hpwl_original/1000.0:.2f}")
    else:
        print("❌ 原始方法HPWL提取失败")
    
    # 对比结果
    if hpwl is not None and hpwl_original is not None:
        print(f"\n=== 结果对比 ===")
        print(f"ISPD2005方法: {hpwl:.0f}")
        print(f"原始方法: {hpwl_original:.0f}")
        print(f"差异: {abs(hpwl - hpwl_original):.0f}")
        print(f"差异百分比: {abs(hpwl - hpwl_original) / max(hpwl, hpwl_original) * 100:.2f}%")
    elif hpwl is not None:
        print(f"\n✅ 只有ISPD2005方法成功: {hpwl:.0f}")
    elif hpwl_original is not None:
        print(f"\n✅ 只有原始方法成功: {hpwl_original:.0f}")
    else:
        print("\n❌ 两种方法都失败")

if __name__ == "__main__":
    test_hpwl_extraction() 