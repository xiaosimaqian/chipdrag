#!/usr/bin/env python3
"""
简化的HPWL提取和布局优化效果测试脚本
验证pin placement功能对HPWL计算的影响
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 直接导入类，避免__init__.py的问题
sys.path.insert(0, str(project_root / "modules" / "rl_training"))

def test_single_design(design_path: str = None):
    """
    测试单个设计的HPWL提取和布局优化效果
    
    Args:
        design_path: 设计目录路径，如果为None则使用默认路径
    """
    # 如果没有指定设计路径，使用默认的ISPD设计
    if design_path is None:
        design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    
    design_dir = Path(design_path)
    if not design_dir.exists():
        print(f"❌ 设计目录不存在: {design_path}")
        return False
    
    print(f"=== 开始测试设计: {design_dir.name} ===")
    print(f"设计路径: {design_dir}")
    
    # 检查设计文件
    required_files = ["design.v", "floorplan.def", "tech.lef", "cells.lef"]
    missing_files = []
    for file_name in required_files:
        if not (design_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        return False
    
    print("✅ 设计文件检查通过")
    
    try:
        # 直接导入类
        from real_openroad_interface_fixed import RealOpenROADInterface
        
        # 创建OpenROAD接口
        print("初始化OpenROAD接口...")
        interface = RealOpenROADInterface(work_dir=str(design_dir))
        
        # 测试1: 基础布局优化（包含pin placement）
        print("\n=== 测试1: 基础布局优化（包含pin placement）===")
        start_time = time.time()
        
        result = interface.run_placement(
            density_target=0.7,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        test1_time = time.time() - start_time
        
        print(f"测试1结果:")
        print(f"  成功: {result['success']}")
        print(f"  执行时间: {result.get('execution_time', test1_time):.2f}秒")
        print(f"  HPWL: {result.get('hpwl', 'N/A')}")
        
        if result['success']:
            print(f"  HPWL值: {result.get('hpwl', float('inf')):.2e}")
        else:
            print(f"  错误: {result.get('stderr', '未知错误')}")
        
        # 测试2: 检查生成的DEF文件
        print("\n=== 测试2: 检查生成的DEF文件 ===")
        def_file = design_dir / "placement_result.def"
        if def_file.exists():
            print(f"✅ DEF文件已生成: {def_file}")
            
            # 检查DEF文件内容
            with open(def_file, 'r') as f:
                def_content = f.read()
            
            # 检查组件放置
            if 'COMPONENTS' in def_content and 'PLACED' in def_content:
                print("✅ DEF文件包含已放置的组件")
                
                # 统计组件数量
                import re
                components_match = re.search(r'COMPONENTS\s+(\d+)', def_content)
                if components_match:
                    num_components = int(components_match.group(1))
                    print(f"  组件总数: {num_components}")
                
                # 统计已放置组件
                placed_count = def_content.count('PLACED')
                print(f"  已放置组件数: {placed_count}")
            else:
                print("❌ DEF文件缺少组件放置信息")
            
            # 检查引脚信息
            if 'PINS' in def_content:
                print("✅ DEF文件包含引脚信息")
                
                # 检查引脚是否有PLACED信息
                if 'PLACED' in def_content and 'LAYER' in def_content:
                    print("✅ 引脚包含PLACED和LAYER信息（pin placement生效）")
                    
                    # 统计引脚数量
                    pins_match = re.search(r'PINS\s+(\d+)', def_content)
                    if pins_match:
                        num_pins = int(pins_match.group(1))
                        print(f"  引脚总数: {num_pins}")
                    
                    # 统计已放置引脚
                    placed_pins = def_content.count('PLACED') - placed_count  # 减去组件的PLACED
                    print(f"  已放置引脚数: {placed_pins}")
                else:
                    print("❌ 引脚缺少PLACED或LAYER信息")
            else:
                print("❌ DEF文件缺少引脚信息")
        else:
            print(f"❌ DEF文件未生成: {def_file}")
        
        # 测试3: 手动提取HPWL验证
        print("\n=== 测试3: 手动提取HPWL验证 ===")
        if def_file.exists():
            hpwl = interface._extract_hpwl_from_def("placement_result.def")
            print(f"手动提取的HPWL: {hpwl:.2e}")
            
            if hpwl != float('inf'):
                print("✅ HPWL提取成功")
            else:
                print("❌ HPWL提取失败")
        else:
            print("❌ 无法提取HPWL：DEF文件不存在")
        
        # 测试4: 检查日志文件
        print("\n=== 测试4: 检查执行日志 ===")
        log_file = design_dir / "openroad_execution.log"
        if log_file.exists():
            print(f"✅ 执行日志已生成: {log_file}")
            
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # 检查pin placement是否执行
            if '引脚布局优化' in log_content:
                print("✅ Pin placement命令已执行")
                if '引脚布局优化完成' in log_content:
                    print("✅ Pin placement执行成功")
                elif '引脚布局优化失败' in log_content:
                    print("❌ Pin placement执行失败")
            else:
                print("❌ 未找到pin placement执行记录")
            
            # 检查布局完成情况
            if '布局完成' in log_content:
                print("✅ 布局流程完成")
            else:
                print("❌ 布局流程未完成")
        else:
            print(f"❌ 执行日志未生成: {log_file}")
        
        # 汇总测试结果
        print(f"\n=== 测试汇总 ===")
        print(f"设计: {design_dir.name}")
        print(f"布局成功: {result['success']}")
        print(f"HPWL: {result.get('hpwl', float('inf')):.2e}")
        print(f"执行时间: {result.get('execution_time', test1_time):.2f}秒")
        print(f"DEF文件生成: {def_file.exists()}")
        print(f"包含组件: {'COMPONENTS' in def_content if def_file.exists() else False}")
        print(f"组件已放置: {'PLACED' in def_content if def_file.exists() else False}")
        print(f"包含引脚: {'PINS' in def_content if def_file.exists() else False}")
        print(f"引脚已放置: {'PLACED' in def_content and 'LAYER' in def_content if def_file.exists() else False}")
        print(f"Pin placement执行: {'引脚布局优化' in log_content if log_file.exists() else False}")
        print(f"Pin placement成功: {'引脚布局优化完成' in log_content if log_file.exists() else False}")
        print(f"布局完成: {'布局完成' in log_content if log_file.exists() else False}")
        
        # 判断测试是否通过
        if (result['success'] and 
            def_file.exists() and 
            'COMPONENTS' in def_content and
            'PLACED' in def_content and
            '引脚布局优化完成' in log_content and
            result.get('hpwl', float('inf')) != float('inf')):
            print("🎉 所有测试项目通过！")
            return True
        else:
            print("⚠️  部分测试项目未通过，请检查上述详细信息")
            return False
        
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 简化HPWL提取和布局优化效果测试 ===")
    
    # 运行测试
    success = test_single_design()
    
    if success:
        print("\n✅ 测试完成")
        return 0
    else:
        print("\n❌ 测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 