#!/usr/bin/env python3
"""
简单的OpenROAD测试脚本 - 验证修复
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入OpenROAD接口
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

def test_single_design():
    """测试单个设计"""
    # 使用一个较小的设计进行测试
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_b"
    
    print(f"=== 测试设计: {design_path} ===")
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface(design_path)
        
        # 生成TCL脚本
        tcl_script = interface._generate_tcl_script()
        
        # 保存TCL脚本
        tcl_file = os.path.join(design_path, "test_openroad_script.tcl")
        with open(tcl_file, 'w') as f:
            f.write(tcl_script)
        
        print(f"TCL脚本已生成: {tcl_file}")
        print("=== TCL脚本内容 ===")
        print(tcl_script)
        
        # 运行OpenROAD
        print("\n=== 运行OpenROAD ===")
        start_time = time.time()
        result = interface.run_placement()
        execution_time = time.time() - start_time
        
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"成功: {result['success']}")
        print(f"返回码: {result['return_code']}")
        
        if result['success']:
            print("✅ 测试成功！")
            print("=== 标准输出 ===")
            for line in result['stdout'][-20:]:  # 只显示最后20行
                print(line)
        else:
            print("❌ 测试失败")
            print("=== 错误输出 ===")
            for line in result['stderr'][-20:]:  # 只显示最后20行
                print(line)
        
        return result['success']
        
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def main():
    """主函数"""
    print("开始OpenROAD修复验证测试...")
    
    success = test_single_design()
    
    if success:
        print("\n🎉 修复验证成功！可以继续批量训练。")
    else:
        print("\n⚠️ 修复验证失败，需要进一步调试。")

if __name__ == "__main__":
    main() 