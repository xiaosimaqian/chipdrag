#!/usr/bin/env python3
"""
增强版OpenROAD接口
支持迭代过程中的布局输出和可视化
"""

import subprocess
import json
import logging
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from modules.parsers.def_parser import DEFParser
from modules.visualization.layout_visualizer import LayoutVisualizer
from matplotlib.animation import FuncAnimation
import hashlib
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedOpenROADInterface:
    """增强版OpenROAD接口，支持迭代布局输出"""
    
    def __init__(self, 
                 docker_image: str = "openroad/flow-ubuntu22.04-builder:21e414",
                 flow_scripts_path: str = "/Users/keqin/Documents/workspace/openroad/OpenROAD-flow-scripts"):
        """初始化增强版OpenROAD接口
        
        Args:
            docker_image: OpenROAD Docker镜像
            flow_scripts_path: OpenROAD流程脚本路径
        """
        self.docker_image = docker_image
        self.flow_scripts_path = flow_scripts_path
        self.iteration_defs = []  # 存储迭代过程中的DEF文件路径
        self.iteration_data = []  # 存储迭代数据
        
        # 设置matplotlib中文字体
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 检查字体是否可用
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            logger.info(f"可用字体: {available_fonts[:5]}...")  # 只显示前5个
            
        except Exception as e:
            logger.warning(f"设置matplotlib字体失败: {str(e)}")
        
    def run_openroad_command(self, 
                           command: str, 
                           work_dir: str,
                           timeout: int = 300) -> Tuple[bool, str, str]:
        """运行OpenROAD命令
        
        Args:
            command: 要执行的命令
            work_dir: 工作目录
            timeout: 超时时间（秒）
            
        Returns:
            Tuple[bool, str, str]: (成功标志, 标准输出, 标准错误)
        """
        try:
            # 确保工作目录是绝对路径
            work_dir = str(Path(work_dir).resolve())
            
            # 构建Docker命令，设置正确的PATH
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{work_dir}:/workspace',
                '-w', '/workspace',
                self.docker_image,
                'bash', '-c', f'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && {command}'
            ]
            
            logger.info(f"执行Docker命令: {' '.join(docker_cmd)}")
            
            # 执行命令
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # 将输出写入本地日志文件
            log_file = Path(work_dir) / "openroad_execution.log"
            with open(log_file, 'w') as f:
                f.write(f"=== OpenROAD Execution Log ===\n")
                f.write(f"Command: {' '.join(docker_cmd)}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write(f"=== STDOUT ===\n")
                f.write(result.stdout)
                f.write(f"\n=== STDERR ===\n")
                f.write(result.stderr)
                f.write(f"\n=== END ===\n")
            
            logger.info(f"OpenROAD执行日志已保存到: {log_file}")
            
            if result.returncode == 0:
                logger.info("OpenROAD命令执行成功")
                return True, result.stdout, result.stderr
            else:
                logger.error(f"OpenROAD命令执行失败: {result.stderr}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"OpenROAD命令执行超时 ({timeout}秒)")
            return False, "", f"命令执行超时 ({timeout}秒)"
        except Exception as e:
            logger.error(f"执行命令失败: {e}")
            return False, "", str(e)
    
    def run_openroad_script_with_monitoring(self, 
                                          script_path: str, 
                                          work_dir: str,
                                          timeout: int = 1800) -> Dict[str, Any]:
        """运行OpenROAD脚本并监控进度
        
        Args:
            script_path: TCL脚本路径
            work_dir: 工作目录
            timeout: 总超时时间（秒）
            
        Returns:
            Dict: 包含执行结果的字典
        """
        try:
            # 确保工作目录是绝对路径
            work_dir = str(Path(work_dir).resolve())
            script_name = Path(script_path).name
            
            # 构建Docker命令
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{work_dir}:/workspace',
                '-w', '/workspace',
                self.docker_image,
                'bash', '-c', f'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -no_splash -exit {script_name}'
            ]
            
            logger.info(f"执行Docker命令: {' '.join(docker_cmd)}")
            
            # 启动进程
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 监控输出
            stdout_lines = []
            stderr_lines = []
            last_output_time = time.time()
            no_output_timeout = 300  # 5分钟无输出超时
            
            while process.poll() is None:
                # 检查总超时
                if time.time() - last_output_time > timeout:
                    logger.error(f"总超时 ({timeout}秒)，终止进程")
                    process.terminate()
                    process.wait(timeout=10)
                    return {
                        "success": False,
                        "error": f"总超时 ({timeout}秒)",
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines)
                    }
                
                # 读取输出
                stdout_line = process.stdout.readline()
                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    last_output_time = time.time()
                    logger.info(f"[进度] 有新日志输出，重置超时计时器。最新输出: {stdout_lines[-3:]}")
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    last_output_time = time.time()
                
                # 检查无输出超时
                if time.time() - last_output_time > no_output_timeout:
                    logger.error(f"超时无新输出，自动终止容器。")
                    process.terminate()
                    process.wait(timeout=10)
                    return {
                        "success": False,
                        "error": f"无输出超时 ({no_output_timeout}秒)",
                        "stdout": "\n".join(stdout_lines),
                        "stderr": "\n".join(stderr_lines)
                    }
                
                time.sleep(1)
            
            # 读取剩余输出
            remaining_stdout, remaining_stderr = process.communicate()
            stdout_lines.extend(remaining_stdout.splitlines())
            stderr_lines.extend(remaining_stderr.splitlines())
            
            # 保存日志
            log_file = Path(work_dir) / "openroad_execution.log"
            with open(log_file, 'w') as f:
                f.write(f"=== OpenROAD Execution Log ===\n")
                f.write(f"Command: {' '.join(docker_cmd)}\n")
                f.write(f"Return Code: {process.returncode}\n")
                f.write(f"=== STDOUT ===\n")
                f.write("\n".join(stdout_lines))
                f.write(f"\n=== STDERR ===\n")
                f.write("\n".join(stderr_lines))
                f.write(f"\n=== END ===\n")
            
            logger.info(f"OpenROAD执行日志已保存到: {log_file}")
            
            if process.returncode == 0:
                logger.info("OpenROAD脚本执行成功")
                return {
                    "success": True,
                    "stdout": "\n".join(stdout_lines),
                    "stderr": "\n".join(stderr_lines)
                }
            else:
                logger.error(f"OpenROAD脚本执行失败: {process.returncode}")
                return {
                    "success": False,
                    "error": f"脚本执行失败 (返回码: {process.returncode})",
                    "stdout": "\n".join(stdout_lines),
                    "stderr": "\n".join(stderr_lines)
                }
                
        except Exception as e:
            logger.error(f"执行OpenROAD脚本失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
    
    def create_iterative_placement_tcl(self, 
                                     verilog_file: str,
                                     lef_files: list,
                                     def_file: str,
                                     work_dir: str,
                                     num_iterations: int = 10) -> str:
        """创建支持迭代输出的Place&Route TCL脚本，自动read所有LEF文件
        
        Args:
            verilog_file: Verilog文件路径
            lef_files: LEF文件路径列表
            def_file: DEF文件路径
            work_dir: 工作目录
            num_iterations: 迭代次数
        """
        # 生成LEF文件读取命令
        lef_read_cmds = []
        for lef in lef_files:
            lef_name = Path(lef).name
            lef_read_cmds.append(f'read_lef "/workspace/{lef_name}"')
        lef_read_cmds_str = "\n".join(lef_read_cmds)
        
        # 获取文件名
        verilog_name = Path(verilog_file).name
        def_name = Path(def_file).name
        
        # 添加详细的调试信息
        logger.info(f"生成TCL脚本 - LEF文件: {lef_files}")
        logger.info(f"生成TCL脚本 - LEF读取命令: {lef_read_cmds_str}")
        logger.info(f"生成TCL脚本 - Verilog文件: {verilog_name}")
        logger.info(f"生成TCL脚本 - DEF文件: {def_name}")
        
        tcl_script = f"""# Enhanced OpenROAD Place & Route Script - RL Training Data Collection
set verilog_file "/workspace/{verilog_name}"
set def_file "/workspace/{def_name}"
set output_dir "/workspace/output"
set num_iterations {num_iterations}

file mkdir $output_dir
file mkdir "$output_dir/iterations"

set log_file "$output_dir/placement_iterations.log"
set log_fp [open $log_file w]

puts "LOG: Current directory: [pwd]"
puts "LOG: output_dir=$output_dir"
puts "LOG: Number of iterations: $num_iterations"
puts "LOG: LEF files to read: {lef_files}"
puts $log_fp "=== OpenROAD RL Training Data Collection ==="
puts $log_fp "Number of iterations: $num_iterations"

# Read LEF files
puts "LOG: Starting to read LEF files..."
{lef_read_cmds_str}
puts "LOG: Completed read_lef"
puts "LOG: Starting read_verilog"
read_verilog $verilog_file
puts "LOG: Completed read_verilog"
puts "LOG: Starting read_def"
read_def $def_file
puts "LOG: Completed read_def"

# Save initial layout (before unplace_all)
puts "LOG: Saving initial layout before unplace_all"
write_def "$output_dir/iterations/iteration_0_initial.def"
puts $log_fp "Iteration 0 (initial): Layout saved before unplace_all"

# Unplace all cells to start fresh for RL training
puts "LOG: Unplacing all cells for RL training"
unplace_all
puts "LOG: All cells unplaced, starting RL training iterations"

# Execute multiple global_placement iterations for RL training
for {{set i 1}} {{$i <= $num_iterations}} {{incr i}} {{
    puts "LOG: Starting RL training iteration $i"
    puts $log_fp "=== RL Training Iteration $i ==="
    
    # Use different placement strategies to create diverse training data
    if {{$i == 1}} {{
        # First iteration: Conservative placement (low density)
        puts "LOG: RL Strategy 1 - Conservative placement"
        global_placement -density 0.70 -wirelength_weight 1.0 -density_weight 0.5
    }} elseif {{$i == 2}} {{
        # Second iteration: Aggressive placement (high density)
        puts "LOG: RL Strategy 2 - Aggressive placement"
        global_placement -density 0.95 -wirelength_weight 0.5 -density_weight 2.0
    }} elseif {{$i == 3}} {{
        # Third iteration: Wirelength-focused placement
        puts "LOG: RL Strategy 3 - Wirelength-focused"
        global_placement -density 0.85 -wirelength_weight 3.0 -density_weight 0.3
    }} elseif {{$i == 4}} {{
        # Fourth iteration: Density-focused placement
        puts "LOG: RL Strategy 4 - Density-focused"
        global_placement -density 0.90 -wirelength_weight 0.3 -density_weight 3.0
    }} elseif {{$i == 5}} {{
        # Fifth iteration: Balanced placement
        puts "LOG: RL Strategy 5 - Balanced placement"
        global_placement -density 0.88 -wirelength_weight 1.5 -density_weight 1.0
    }} elseif {{$i == 6}} {{
        # Sixth iteration: High wirelength weight
        puts "LOG: RL Strategy 6 - High wirelength weight"
        global_placement -density 0.82 -wirelength_weight 4.0 -density_weight 0.2
    }} elseif {{$i == 7}} {{
        # Seventh iteration: High density weight
        puts "LOG: RL Strategy 7 - High density weight"
        global_placement -density 0.92 -wirelength_weight 0.2 -density_weight 4.0
    }} elseif {{$i == 8}} {{
        # Eighth iteration: Mixed strategy
        puts "LOG: RL Strategy 8 - Mixed strategy"
        global_placement -density 0.87 -wirelength_weight 2.0 -density_weight 1.5
    }} elseif {{$i == 9}} {{
        # Ninth iteration: Extreme wirelength optimization
        puts "LOG: RL Strategy 9 - Extreme wirelength optimization"
        global_placement -density 0.75 -wirelength_weight 5.0 -density_weight 0.1
    }} else {{
        # Tenth iteration: Final balanced optimization
        puts "LOG: RL Strategy 10 - Final balanced optimization"
        global_placement -density 0.89 -wirelength_weight 1.0 -density_weight 1.0
    }}
    
    # Save current layout for RL training data
    set def_filename "$output_dir/iterations/iteration_${{i}}_rl_training.def"
    write_def $def_filename
    puts "LOG: RL training layout saved to: $def_filename"
    puts $log_fp "DEF file: $def_filename"
    
    # Collect comprehensive metrics for RL reward calculation
    puts "LOG: Collecting RL training metrics"
    
    # Get HPWL information for wirelength reward
    set hpwl_report_file "$output_dir/iterations/iteration_${{i}}_hpwl.rpt"
    if {{[catch {{report_wire_length -net}} result]}} {{
        puts "LOG: Cannot get HPWL information: $result"
        puts $log_fp "Iteration $i: HPWL=unavailable"
    }} else {{
        set hpwl_fp [open $hpwl_report_file w]
        puts $hpwl_fp $result
        close $hpwl_fp
        puts "LOG: HPWL report saved to: $hpwl_report_file"
        puts $log_fp "Iteration $i: HPWL report saved"
    }}
    
    # Get overflow information for density reward
    set overflow_report_file "$output_dir/iterations/iteration_${{i}}_overflow.rpt"
    if {{[catch {{report_placement_overflow}} result]}} {{
        puts "LOG: Cannot get overflow information: $result"
        puts $log_fp "Iteration $i: Overflow=unavailable"
    }} else {{
        set overflow_fp [open $overflow_report_file w]
        puts $overflow_fp $result
        close $overflow_fp
        puts "LOG: Overflow report saved to: $overflow_report_file"
        puts $log_fp "Iteration $i: Overflow report saved"
    }}
    
    # Get timing information for timing reward (if available)
    set timing_report_file "$output_dir/iterations/iteration_${{i}}_timing.rpt"
    if {{[catch {{report_timing}} result]}} {{
        puts "LOG: Cannot get timing information: $result"
        puts $log_fp "Iteration $i: Timing=unavailable"
    }} else {{
        set timing_fp [open $timing_report_file w]
        puts $timing_fp $result
        close $timing_fp
        puts "LOG: Timing report saved to: $timing_report_file"
        puts $log_fp "Iteration $i: Timing report saved"
    }}
    
    puts $log_fp "---"
}}

# Generate final area report for RL training evaluation
puts "LOG: Generating RL training evaluation report"
set area_fp [open "$output_dir/rl_training_evaluation.rpt" w]

# Get Die and Core area
set die_area [ord::get_die_area]
set core_area [ord::get_core_area]

if {{[llength $die_area] == 4}} {{
    set die_w [expr {{[lindex $die_area 2] - [lindex $die_area 0]}}]
    set die_h [expr {{[lindex $die_area 3] - [lindex $die_area 1]}}]
    set die_size [expr {{$die_w * $die_h}}]
    puts $area_fp "Die area: $die_size (width: $die_w, height: $die_h)"
    puts $area_fp "Die coordinates: $die_area"
}} else {{
    puts $area_fp "Die area: unavailable (coordinates: $die_area)"
}}

if {{[llength $core_area] == 4}} {{
    set core_w [expr {{[lindex $core_area 2] - [lindex $core_area 0]}}]
    set core_h [expr {{[lindex $core_area 3] - [lindex $core_area 1]}}]
    set core_size [expr {{$core_w * $core_h}}]
    puts $area_fp "Core area: $core_size (width: $core_w, height: $core_h)"
    puts $area_fp "Core coordinates: $core_area"
}} else {{
    puts $area_fp "Core area: unavailable (coordinates: $core_area)"
}}

# Calculate utilization for RL reward
if {{[info exists die_size] && [info exists core_size] && $die_size > 0}} {{
    set layout_utilization [expr {{double($core_size) / double($die_size) * 100.0}}]
    puts $area_fp "Layout utilization (Core area/Die area): $layout_utilization%"
}}

close $area_fp

# Close log file
puts $log_fp "=== RL Training Data Collection Completed ==="
close $log_fp

puts "LOG: RL training data collection completed, log saved to: $log_file"
"""
        
        # 保存TCL脚本到工作目录
        script_path = Path(work_dir) / "iterative_placement.tcl"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        logger.info(f"TCL脚本已保存到: {script_path}")
        return str(script_path)
    
    def run_iterative_placement(self, 
                              work_dir: str,
                              num_iterations: int = 10) -> Dict[str, Any]:
        """运行迭代布局流程，自动收集LEF、Verilog和DEF文件，并检查LEF/LIB文件
        
        Args:
            work_dir: 工作目录
            num_iterations: 迭代次数
        """
        # 自动收集所有LEF文件
        lef_files = list(Path(work_dir).glob("*.lef"))
        lef_files = [str(f) for f in lef_files]
        if not lef_files:
            logger.error(f"未找到LEF文件: {work_dir}")
            return {"success": False, "error": "未找到LEF文件"}
        else:
            logger.info(f"找到LEF文件: {lef_files}")
        
        # 检查LIB文件（可选）
        lib_files = list(Path(work_dir).glob("*.lib"))
        lib_files = [str(f) for f in lib_files]
        if not lib_files:
            logger.warning(f"未找到LIB文件: {work_dir}，部分功能如时序/功耗分析可能不可用")
        else:
            logger.info(f"找到LIB文件: {lib_files}")
        
        # 自动收集唯一的Verilog文件
        v_files = list(Path(work_dir).glob("*.v"))
        v_files = [str(f) for f in v_files]
        if len(v_files) == 0:
            logger.error(f"未找到Verilog文件: {work_dir}")
            return {"success": False, "error": "未找到Verilog文件"}
        if len(v_files) > 1:
            logger.error(f"找到多个Verilog文件，请手动指定: {v_files}")
            return {"success": False, "error": f"找到多个Verilog文件: {v_files}"}
        verilog_file = v_files[0]
        
        # 自动收集唯一的DEF文件
        def_files = list(Path(work_dir).glob("*.def"))
        def_files = [str(f) for f in def_files]
        if len(def_files) == 0:
            logger.error(f"未找到DEF文件: {work_dir}")
            return {"success": False, "error": "未找到DEF文件"}
        
        # 智能选择初始布局文件
        def_file = None
        priority_keywords = ["floorplan", "initial", "base", "start"]
        
        # 按优先级查找初始布局文件
        for keyword in priority_keywords:
            for df in def_files:
                if keyword in Path(df).name.lower():
                    def_file = df
                    logger.info(f"找到初始布局文件 (关键词 '{keyword}'): {def_file}")
                    break
            if def_file:
                break
        
        # 如果没有找到合适的初始文件，使用第一个DEF文件
        if def_file is None:
            if len(def_files) > 1:
                logger.warning(f"找到多个DEF文件，使用第一个作为初始布局: {def_files[0]}")
                logger.info(f"可用DEF文件: {[Path(f).name for f in def_files]}")
            def_file = def_files[0]
        
        logger.info(f"使用DEF文件作为初始布局: {def_file}")
        
        tcl_content = self.create_iterative_placement_tcl(
            verilog_file=verilog_file,
            lef_files=lef_files,
            def_file=def_file,
            work_dir=work_dir,
            num_iterations=num_iterations
        )
        
        try:
            logger.info(f"开始运行迭代Place&Route流程，迭代次数: {num_iterations}")
            
            # 运行OpenROAD命令
            command = f"openroad -exit iterative_placement.tcl"
            success, stdout, stderr = self.run_openroad_command(command, work_dir)
            
            if not success:
                logger.error(f"迭代Place&Route流程失败: {stderr}")
                return {"success": False, "error": stderr}
            
            # 收集迭代数据
            iteration_data = self.collect_iteration_data(f"{work_dir}/output/iterations")
            
            # 生成可视化
            visualization_path = self.generate_iteration_visualization(
                iteration_data, f"{work_dir}/output"
            )
            
            return {
                "success": True,
                "iteration_data": iteration_data,
                "visualization_path": visualization_path,
                "output_dir": f"{work_dir}/output"
            }
            
        except Exception as e:
            logger.error(f"迭代Place&Route流程异常: {e}")
            return {"success": False, "error": str(e)}
    
    def collect_iteration_data(self, iterations_dir: str) -> List[Dict[str, Any]]:
        """收集迭代数据
        
        Args:
            iterations_dir: 迭代DEF文件目录
            
        Returns:
            List[Dict[str, Any]]: 迭代数据列表
        """
        
        iteration_data = []
        iterations_path = Path(iterations_dir)
        
        if not iterations_path.exists():
            logger.warning(f"迭代目录不存在: {iterations_dir}")
            return iteration_data
        
        # 查找所有DEF文件
        def_files = list(iterations_path.glob("*.def"))
        def_files.sort(key=lambda x: x.name)
        
        for i, def_file in enumerate(def_files):
            # 解析文件名获取迭代信息
            filename = def_file.name
            
            # 处理初始布局
            if "initial" in filename:
                iteration_num = 0
            else:
                # 从文件名中提取迭代号
                import re
                match = re.search(r'iteration_(\d+)', filename)
                if match:
                    iteration_num = int(match.group(1))
                else:
                    iteration_num = i
            
            # 获取HPWL和溢出率
            hpwl = None
            overflow = None
            
            # 尝试读取HPWL报告文件
            hpwl_report_file = iterations_path / f"iteration_{iteration_num}_hpwl.rpt"
            if hpwl_report_file.exists():
                try:
                    with open(hpwl_report_file, 'r') as f:
                        hpwl_content = f.read()
                    
                    # 解析HPWL报告内容
                    # OpenROAD的report_wire_length输出格式通常是：
                    # "Total wire length: XXXX um"
                    import re
                    hpwl_match = re.search(r'Total wire length:\s*([\d.]+)\s*um', hpwl_content)
                    if hpwl_match:
                        hpwl = float(hpwl_match.group(1))
                        logger.info(f"从HPWL报告解析到值: {hpwl}")
                    else:
                        logger.warning(f"无法从HPWL报告解析数值: {hpwl_report_file}")
                        
                except Exception as e:
                    logger.warning(f"读取HPWL报告失败: {str(e)}")
            
            # 尝试读取溢出率报告文件
            overflow_report_file = iterations_path / f"iteration_{iteration_num}_overflow.rpt"
            if overflow_report_file.exists():
                try:
                    with open(overflow_report_file, 'r') as f:
                        overflow_content = f.read()
                    
                    # 解析溢出率报告内容
                    # OpenROAD的report_placement_overflow输出格式通常是：
                    # "Overflow: XXXX%"
                    import re
                    overflow_match = re.search(r'Overflow:\s*([\d.]+)\s*%', overflow_content)
                    if overflow_match:
                        overflow = float(overflow_match.group(1)) / 100.0  # 转换为小数
                        logger.info(f"从溢出率报告解析到值: {overflow}")
                    else:
                        logger.warning(f"无法从溢出率报告解析数值: {overflow_report_file}")
                        
                except Exception as e:
                    logger.warning(f"读取溢出率报告失败: {str(e)}")
            
            iteration_info = {
                'iteration': iteration_num,
                'def_file': str(def_file),
                'hpwl': hpwl,
                'overflow': overflow,
                'timestamp': def_file.stat().st_mtime if def_file.exists() else None
            }
            
            iteration_data.append(iteration_info)
            
            logger.info(f"迭代 {iteration_num}: HPWL={hpwl}, 溢出率={overflow}")
        
        return iteration_data
    
    def extract_stats_from_def(self, def_file: str) -> Tuple[Optional[float], Optional[float]]:
        """从DEF文件中提取HPWL和溢出率信息
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (hpwl, overflow)
        """
        try:
            # 这里可以添加从DEF文件中解析统计信息的逻辑
            # 目前返回None，后续可以通过解析DEF文件内容来获取
            return None, None
        except Exception as e:
            logger.warning(f"从DEF文件提取统计信息失败: {e}")
            return None, None
    
    def generate_iteration_visualization(self, iteration_data: List[Dict[str, Any]], 
                                       output_dir: str) -> str:
        """生成迭代可视化图表
        
        Args:
            iteration_data: 迭代数据列表
            output_dir: 输出目录
            
        Returns:
            str: 可视化文件路径
        """
        try:
            logger.info("开始生成迭代可视化")
            
            # 创建可视化目录
            viz_dir = Path(output_dir) / "visualization"
            viz_dir.mkdir(exist_ok=True)
            
            # 生成指标演化图
            if len(iteration_data) > 1:
                self.plot_metrics_evolution(iteration_data, viz_dir)
            
            # 生成布局对比图
            self.plot_layout_comparison(iteration_data, viz_dir)
            
            # 生成布局演化动画
            self.create_layout_animation(iteration_data, viz_dir)
            
            logger.info(f"可视化文件保存在: {viz_dir}")
            return str(viz_dir)
            
        except Exception as e:
            logger.error(f"生成迭代可视化失败: {str(e)}")
            return ""
    
    def plot_metrics_evolution(self, 
                             iteration_data: List[Dict[str, Any]], 
                             output_dir: Path):
        """绘制指标演化图"""
        
        # 过滤有效数据
        valid_data = [d for d in iteration_data if d["hpwl"] is not None and d["overflow"] is not None]
        
        if len(valid_data) < 2:
            logger.warning("有效数据不足，跳过指标演化图")
            return
        
        iterations = [d["iteration"] for d in valid_data]
        hpwl_values = [d["hpwl"] for d in valid_data]
        overflow_values = [d["overflow"] for d in valid_data]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # HPWL变化图
        ax1.plot(iterations, hpwl_values, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('HPWL (um)')
        ax1.set_title('HPWL演化过程')
        ax1.grid(True, alpha=0.3)
        
        # 溢出率变化图
        ax2.plot(iterations, overflow_values, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('溢出率')
        ax2.set_title('溢出率演化过程')
        ax2.grid(True, alpha=0.3)
        
        # 添加收敛线
        ax2.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='收敛阈值 (0.1)')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图片
        metrics_path = output_dir / "metrics_evolution.png"
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"指标演化图已保存: {metrics_path}")
    
    def plot_layout_comparison(self, iteration_data: List[Dict[str, Any]], 
                             viz_dir: Path) -> None:
        """绘制布局对比图"""
        try:
            # 计算子图数量
            num_plots = len(iteration_data)
            if num_plots == 0:
                logger.warning("没有迭代数据可绘制")
                return
            
            # 计算子图布局
            cols = min(4, num_plots)
            rows = (num_plots + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if num_plots == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, data in enumerate(iteration_data):
                if i >= len(axes):
                    break
                    
                def_file = data.get('def_file', '')
                iter_idx = data.get('iteration', i)
                
                if def_file and Path(def_file).exists():
                    try:
                        # 计算DEF文件的MD5哈希值用于诊断
                        content_bytes = Path(def_file).read_bytes()
                        file_hash = hashlib.md5(content_bytes).hexdigest()
                        logger.info(f"迭代 {iter_idx}: DEF='{Path(def_file).name}', MD5='{file_hash}'")
                        
                        parser = DEFParser()
                        with open(def_file, 'r') as f:
                            def_content = f.read()
                        layout_data = parser.parse_def(def_content)
                        
                        self.plot_layout_real(layout_data, axes[i], f"迭代 {iter_idx}")
                    except Exception as e:
                        logger.warning(f"迭代 {iter_idx} 解析失败: {str(e)}")
                        axes[i].text(0.5, 0.5, f"迭代 {iter_idx}\n解析失败", 
                                   ha='center', va='center', transform=axes[i].transAxes)
                else:
                    axes[i].text(0.5, 0.5, f"迭代 {iter_idx}\n文件不存在", 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "layout_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"绘制布局对比图失败: {str(e)}")
    
    def create_layout_animation(self, iteration_data: List[Dict[str, Any]], 
                              viz_dir: Path) -> None:
        """创建布局演化动画"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            def animate(frame):
                ax.clear()
                if frame < len(iteration_data):
                    data = iteration_data[frame]
                    def_file = data.get('def_file', '')
                    iter_idx = data.get('iteration', frame)
                    
                    if def_file and Path(def_file).exists():
                        try:
                            # 计算DEF文件的MD5哈希值用于诊断
                            content_bytes = Path(def_file).read_bytes()
                            file_hash = hashlib.md5(content_bytes).hexdigest()
                            logger.info(f"动画帧 {frame} (迭代 {iter_idx}): DEF='{Path(def_file).name}', MD5='{file_hash}'")
                            
                            parser = DEFParser()
                            with open(def_file, 'r') as f:
                                def_content = f.read()
                            layout_data = parser.parse_def(def_content)
                            
                            self.plot_layout_real(layout_data, ax, f"迭代 {iter_idx}")
                        except Exception as e:
                            logger.warning(f"动画帧 {frame} 解析失败: {str(e)}")
                            ax.text(0.5, 0.5, f"迭代 {iter_idx}\n解析失败", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, f"迭代 {iter_idx}\n文件不存在", 
                               ha='center', va='center', transform=ax.transAxes)
            
            anim = FuncAnimation(fig, animate, frames=len(iteration_data), 
                               interval=500, repeat=True)
            anim.save(viz_dir / "layout_evolution.gif", writer='pillow')
            plt.close()
            
        except Exception as e:
            logger.error(f"创建布局动画失败: {str(e)}")
    
    def plot_layout_real(self, layout_data: Dict[str, Any], ax, title: str = "布局图") -> None:
        """绘制真实布局图
        
        Args:
            layout_data: 解析后的DEF数据
            ax: matplotlib轴对象
            title: 图表标题
        """
        try:
            logger.info(f"开始绘制布局图: {title}")
            logger.info(f"布局数据键: {list(layout_data.keys())}")
            
            # 获取Die区域
            die_area = layout_data.get('DIEAREA', {})
            logger.info(f"Die区域数据: {die_area}")
            
            if die_area and isinstance(die_area, dict):
                # 尝试不同的坐标格式
                coords = []
                
                # 格式1: coordinates数组
                if 'coordinates' in die_area:
                    coords = die_area['coordinates']
                # 格式2: 直接坐标字段
                elif 'x1' in die_area and 'y1' in die_area and 'x2' in die_area and 'y2' in die_area:
                    coords = [die_area['x1'], die_area['y1'], die_area['x2'], die_area['y2']]
                # 格式3: 其他可能的字段名
                elif 'x' in die_area and 'y' in die_area:
                    if isinstance(die_area['x'], list) and isinstance(die_area['y'], list):
                        coords = [die_area['x'][0], die_area['y'][0], die_area['x'][1], die_area['y'][1]]
                
                logger.info(f"Die坐标: {coords}")
                
                if coords and len(coords) >= 4:
                    x_coords = [float(coords[0]), float(coords[2])]
                    y_coords = [float(coords[1]), float(coords[3])]
                    
                    # 绘制Die边界
                    ax.plot([x_coords[0], x_coords[1], x_coords[1], x_coords[0], x_coords[0]], 
                           [y_coords[0], y_coords[0], y_coords[1], y_coords[1], y_coords[0]], 
                           'k-', linewidth=2, label='Die边界')
                    
                    # 设置坐标轴范围
                    ax.set_xlim(x_coords[0] - 1000, x_coords[1] + 1000)
                    ax.set_ylim(y_coords[0] - 1000, y_coords[1] + 1000)
                    
                    logger.info(f"Die边界绘制完成: ({x_coords[0]}, {y_coords[0]}) -> ({x_coords[1]}, {y_coords[1]})")
            
            # 绘制行
            rows = layout_data.get('ROWS', [])
            if rows and isinstance(rows, list):
                row_x, row_y = [], []
                # 假设每个row是一个点
                for row_data in rows:
                    if 'x' in row_data and 'y' in row_data:
                        row_x.append(row_data['x'])
                        row_y.append(row_data['y'])
                if row_x:
                    ax.scatter(row_x, row_y, s=5, color='green', label=f'行 ({len(row_x)}个)')
            else:
                logger.warning("在布局数据中没有找到'ROWS'或格式不正确")

            # 绘制组件
            components = layout_data.get('COMPONENTS', [])
            logger.info(f"找到 {len(components)} 个组件用于绘制")
            if components and isinstance(components, list) and components:
                comp_x, comp_y = [], []
                for comp_data in components:
                    if comp_data.get('status') in ['PLACED', 'FIXED']:
                        if 'x' in comp_data and 'y' in comp_data:
                            comp_x.append(comp_data['x'])
                            comp_y.append(comp_data['y'])
                
                if comp_x:
                    ax.scatter(comp_x, comp_y, s=1, color='blue', alpha=0.5, label=f'单元 ({len(comp_x)}个)')
                else:
                    logger.warning("有组件数据，但没有找到有效的坐标信息")
                    ax.text(0.5, 0.5, '无已放置的组件', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
            else:
                logger.warning("在布局数据中没有找到'COMPONENTS'段或'COMPONENTS'段为空")
                ax.text(0.5, 0.5, '无组件信息', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X坐标 (um)")
            ax.set_ylabel("Y坐标 (um)")
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            logger.info(f"布局图绘制完成: {title}")
            
        except Exception as e:
            logger.error(f"绘制布局图失败: {str(e)}")
            # 绘制错误信息
            ax.text(0.5, 0.5, f"绘制失败\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            ax.set_title(title, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

def test_enhanced_openroad():
    """测试增强版OpenROAD接口"""
    
    # 设计目录
    design_dir = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1"
    
    # 文件路径
    verilog_file = f"{design_dir}/design.v"
    cells_lef = f"{design_dir}/cells.lef"
    tech_lef = f"{design_dir}/tech.lef"
    def_file = f"{design_dir}/mgc_des_perf_1_place.def"
    constraints_file = f"{design_dir}/constraints.sdc"
    
    # 检查文件是否存在
    for file_path in [verilog_file, cells_lef, tech_lef, def_file]:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return
    
    # 创建接口实例
    interface = EnhancedOpenROADInterface()
    
    # 运行迭代Place&Route
    result = interface.run_iterative_placement(
        work_dir=design_dir,
        num_iterations=10  # 迭代次数
    )
    
    if result["success"]:
        logger.info("迭代Place&Route成功完成")
        logger.info(f"可视化路径: {result['visualization_path']}")
        
        # 输出迭代数据
        iteration_data = result["iteration_data"]
        logger.info(f"总共收集到 {len(iteration_data)} 个迭代数据")
        
        for data in iteration_data:
            logger.info(f"迭代 {data['iteration']}: HPWL={data.get('hpwl', 'N/A')}, "
                       f"溢出率={data.get('overflow', 'N/A')}")
    else:
        logger.error(f"迭代Place&Route失败: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    test_enhanced_openroad() 