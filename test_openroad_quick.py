#!/usr/bin/env python3
"""
OpenROAD快速测试脚本
使用最小的设计进行快速验证
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

def test_smallest_design():
    """测试最小的设计"""
    # 使用最小的设计进行测试
    design_path = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_b"
    
    print(f"=== 快速测试最小设计: {design_path} ===")
    
    try:
        # 创建接口实例
        interface = RealOpenROADInterface(design_path)
        
        # 生成简化的TCL脚本
        tcl_script = """# OpenROAD快速测试脚本
# 最小化配置，快速验证

# 完全重置数据库
if {[info exists ::ord::db]} {
    ord::reset_db
}

# 读取LEF文件
read_lef tech.lef
read_lef cells.lef

# 读取Verilog文件
read_verilog design.v

# 连接设计
link_design des_perf

# 快速初始化布局 - 使用更大的区域
puts "快速初始化布局..."
# 根据实例数量计算合适的区域大小
# 112644个实例，每个实例平均1.6 um²，需要约180,000 um²
# 加上一些余量，使用600x600的区域
initialize_floorplan -die_area "0 0 600 600" -core_area "10 10 590 590" -site core

# 快速全局布局
puts "快速全局布局..."
global_placement -density 0.7

# 快速详细布局
puts "快速详细布局..."
detailed_placement

# 输出结果
write_def quick_test_result.def
write_verilog quick_test_result.v

puts "快速测试完成"
puts "输出文件: quick_test_result.def, quick_test_result.v"
"""
        
        # 保存TCL脚本
        tcl_file = os.path.join(design_path, "quick_test_script.tcl")
        with open(tcl_file, 'w') as f:
            f.write(tcl_script)
        
        print(f"快速TCL脚本已生成: {tcl_file}")
        print("=== TCL脚本内容 ===")
        print(tcl_script)
        
        # 直接运行Docker命令
        print("\n=== 运行快速OpenROAD测试 ===")
        start_time = time.time()
        
        work_dir_abs = Path(design_path).resolve()
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir_abs}:/workspace",
            "-w", "/workspace",
            "openroad/flow-ubuntu22.04-builder:21e414",
            "bash", "-c",
            "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit quick_test_script.tcl"
        ]
        
        import subprocess
        try:
            # 使用2分钟超时
            result = subprocess.run(
                docker_cmd, 
                capture_output=True, 
                text=True, 
                timeout=120  # 2分钟超时
            )
            execution_time = time.time() - start_time
            
            print(f"执行时间: {execution_time:.2f}秒")
            print(f"返回码: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ 快速测试成功！")
                print("=== 标准输出 ===")
                for line in result.stdout.split('\n')[-10:]:  # 只显示最后10行
                    if line.strip():
                        print(line)
            else:
                print("❌ 快速测试失败")
                print("=== 错误输出 ===")
                for line in result.stderr.split('\n')[-10:]:  # 只显示最后10行
                    if line.strip():
                        print(line)
            
            # 检查输出文件
            result_def = os.path.join(design_path, "quick_test_result.def")
            result_verilog = os.path.join(design_path, "quick_test_result.v")
            
            if os.path.exists(result_def):
                print(f"✅ 生成DEF文件: {result_def}")
            if os.path.exists(result_verilog):
                print(f"✅ 生成Verilog文件: {result_verilog}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print("❌ 快速测试超时（2分钟）")
            return False
            
    except Exception as e:
        print(f"❌ 快速测试异常: {e}")
        return False

def main():
    """主函数"""
    print("开始OpenROAD快速验证测试...")
    
    success = test_smallest_design()
    
    if success:
        print("\n🎉 快速验证成功！OpenROAD环境正常。")
        print("现在可以运行完整的批量训练了。")
    else:
        print("\n⚠️ 快速验证失败，需要进一步调试。")

if __name__ == "__main__":
    main() 