#!/usr/bin/env python3
"""
简化OpenROAD测试 - 只进行基本文件读取验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
import json

def test_simple_openroad():
    """简化OpenROAD测试"""
    print("=== 简化OpenROAD测试 ===")
    
    # 初始化接口
    try:
        interface = RealOpenROADInterface()
        print("✓ 接口初始化成功")
    except Exception as e:
        print(f"✗ 接口初始化失败: {e}")
        return
    
    # 生成简化的TCL脚本
    simple_tcl = """
# 简化OpenROAD测试脚本
# 只进行基本文件读取验证

# 重置数据库
if {[info exists ::ord::db]} {
    ord::reset_db
}

# 读取LEF文件
read_lef -tech tech.lef
read_lef -library cells.lef

# 读取Verilog文件
read_verilog design.v

# 连接设计
link_design des_perf

# 输出基本信息
puts "OpenROAD基本测试完成"
puts "设计名称: [get_db current_design .name]"
puts "单元数量: [llength [get_db insts]]"

exit
"""
    
    # 写入TCL文件
    tcl_file = interface.work_dir / "simple_test.tcl"
    with open(tcl_file, 'w') as f:
        f.write(simple_tcl)
    
    print(f"✓ 简化TCL脚本已生成: {tcl_file}")
    
    # 构建Docker命令
    work_dir_abs = interface.work_dir.resolve()
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{work_dir_abs}:/workspace",
        "-w", "/workspace",
        "openroad/flow-ubuntu22.04-builder:21e414",
        "bash", "-c",
        f"export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit simple_test.tcl"
    ]
    
    print(f"执行Docker命令: {' '.join(docker_cmd)}")
    
    # 执行命令
    import subprocess
    import time
    
    start_time = time.time()
    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=60  # 1分钟超时
    )
    execution_time = time.time() - start_time
    
    print(f"执行时间: {execution_time:.2f}秒")
    print(f"返回码: {result.returncode}")
    print(f"成功: {result.returncode == 0}")
    
    if result.stdout:
        print("=== 标准输出 ===")
        for line in result.stdout.split('\n')[:20]:  # 只显示前20行
            print(f"  {line}")
    
    if result.stderr:
        print("=== 错误输出 ===")
        for line in result.stderr.split('\n')[:10]:  # 只显示前10行
            print(f"  {line}")
    
    # 清理文件
    if tcl_file.exists():
        tcl_file.unlink()
    
    return result.returncode == 0

if __name__ == "__main__":
    success = test_simple_openroad()
    print(f"\n测试结果: {'成功' if success else '失败'}") 