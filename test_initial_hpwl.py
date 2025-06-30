#!/usr/bin/env python3
"""测试初始布局的HPWL提取"""

import subprocess
import tempfile
import os

def test_initial_hpwl():
    """测试初始布局的HPWL提取"""
    design_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_a"
    def_file = "output/iterations/iteration_0_initial.def"
    
    # 创建临时TCL脚本
    tcl_content = f"""
read_def {def_file}
report_wirelength
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
        f.write(tcl_content)
        tcl_file = f.name
    
    try:
        # 运行OpenROAD命令
        cmd = f"docker run --rm -v {os.path.abspath(design_dir)}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit {os.path.basename(tcl_file)}'"
        
        print(f"执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        print(f"返回码: {result.returncode}")
        print(f"标准输出:\n{result.stdout}")
        print(f"标准错误:\n{result.stderr}")
        
        # 分析输出
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'HPWL' in line or 'wirelength' in line.lower():
                    print(f"找到HPWL相关行: {line}")
        else:
            print("OpenROAD命令执行失败")
            
    finally:
        # 清理临时文件
        os.unlink(tcl_file)

if __name__ == "__main__":
    test_initial_hpwl() 