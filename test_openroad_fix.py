#!/usr/bin/env python3
"""
测试OpenROAD脚本修复
验证"Chip already exists"问题是否解决
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from paper_hpwl_comparison_experiment_fixed import PaperHPWLComparisonExperimentFixed

def test_openroad_script_fix():
    """测试OpenROAD脚本修复"""
    
    print("🧪 测试OpenROAD脚本修复...")
    
    # 初始化实验
    experiment = PaperHPWLComparisonExperimentFixed()
    
    # 选择一个小设计进行测试
    test_design = "mgc_fft_1"
    design_dir = Path(f"dataset/ispd_2015_contest_benchmark/{test_design}")
    
    if not design_dir.exists():
        print(f"❌ 测试设计目录不存在: {design_dir}")
        return False
        
    print(f"✅ 找到测试设计: {test_design}")
    
    # 检查必要文件
    required_files = ["tech.lef", "cells.lef", "floorplan.def", "design.v"]
    for file_name in required_files:
        file_path = design_dir / file_name
        if not file_path.exists():
            print(f"❌ 缺少必要文件: {file_path}")
            return False
    
    print("✅ 所有必要文件都存在")
    
    # 生成修复后的脚本
    layout_strategy = {
        'utilization': 0.7,
        'aspect_ratio': 1.0
    }
    
    try:
        script_content = experiment._generate_openroad_script(layout_strategy, test_design)
        print("✅ 成功生成修复后的OpenROAD脚本")
        
        # 保存脚本用于检查
        script_file = design_dir / "run_placement_fixed.tcl"
        with open(script_file, 'w') as f:
            f.write(script_content)
        print(f"✅ 脚本已保存到: {script_file}")
        
        # 尝试执行一次测试
        print("🚀 尝试执行OpenROAD测试...")
        success = experiment._execute_openroad_layout(design_dir, layout_strategy)
        
        if success:
            print("🎉 OpenROAD执行成功！'Chip already exists'问题已修复")
            return True
        else:
            print("⚠️ OpenROAD执行失败，需要进一步检查")
            
            # 检查错误日志
            error_log = design_dir / "error.log"
            if error_log.exists():
                print(f"\n📋 错误日志内容 ({error_log}):")
                with open(error_log, 'r') as f:
                    lines = f.readlines()
                    # 显示最后20行
                    for line in lines[-20:]:
                        if "ERROR" in line or "❌" in line:
                            print(f"🔍 {line.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = test_openroad_script_fix()
    
    if success:
        print("\n🎉 测试通过！OpenROAD脚本修复成功")
        exit(0)
    else:
        print("\n❌ 测试失败，需要进一步调试")
        exit(1) 