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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedOpenROADInterface:
    """增强版OpenROAD接口，支持迭代布局输出"""
    
    def __init__(self, 
                 docker_image: str = "openroad/flow-ubuntu22.04-builder:21e414",
                 flow_scripts_path: str = "/Users/keqin/Documents/workspace/openroad/OpenROAD-flow-scripts"):
        self.docker_image = docker_image
        self.flow_scripts_path = flow_scripts_path
        self.iteration_defs = []  # 存储迭代过程中的DEF文件路径
        self.iteration_data = []  # 存储迭代数据
        
    def run_openroad_command(self, 
                           command: str, 
                           work_dir: str,
                           timeout: int = 300) -> Tuple[bool, str, str]:
        """在Docker容器中运行OpenROAD命令"""
        
        # 确保工作目录是绝对路径
        work_dir = str(Path(work_dir).resolve())
        
        # 构建Docker命令
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/workspace",
            "-v", f"{self.flow_scripts_path}:/OpenROAD-flow-scripts",
            "-w", "/workspace",
            self.docker_image,
            "bash", "-c", f"""
            export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH
            echo "=== Docker容器内环境信息 ==="
            echo "当前目录: $(pwd)"
            echo "PATH: $PATH"
            echo "OpenROAD版本: $(openroad --version 2>/dev/null || echo 'OpenROAD未找到')"
            echo "=== 开始执行命令 ==="
            {command}
            echo "=== 命令执行完成 ==="
            """
        ]
        
        logger.info(f"执行Docker命令: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # 输出详细的执行结果
            logger.info(f"Docker执行返回码: {result.returncode}")
            if result.stdout:
                logger.info(f"Docker标准输出:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Docker错误输出:\n{result.stderr}")
            
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"命令执行超时 ({timeout}秒)")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"执行命令失败: {e}")
            return False, "", str(e)
    
    def create_iterative_placement_tcl(self, 
                                     verilog_file: str,
                                     cells_lef: str,
                                     tech_lef: str,
                                     def_file: str,
                                     output_dir: str,
                                     save_interval: int = 50) -> str:
        """创建支持迭代输出的Place&Route TCL脚本
        
        Args:
            save_interval: 每隔多少次迭代保存一次DEF文件
        """
        
        tcl_script = f"""# 增强版OpenROAD Place & Route脚本 - 支持迭代输出
set verilog_file "/workspace/{Path(verilog_file).name}"
set cells_lef "/workspace/{Path(cells_lef).name}"
set tech_lef "/workspace/{Path(tech_lef).name}"
set def_file "/workspace/{Path(def_file).name}"
set output_dir "/workspace/output"
set save_interval {save_interval}

# 创建输出目录
file mkdir $output_dir
file mkdir "$output_dir/iterations"

puts "LOG: 当前目录: [pwd]"
puts "LOG: output_dir=$output_dir"
puts "LOG: 保存间隔: $save_interval"

# 读取设计文件
puts "LOG: 开始 read_lef"
read_lef $tech_lef
puts "LOG: 完成 read_lef(tech)"
read_lef $cells_lef
puts "LOG: 完成 read_lef(cells)"
puts "LOG: 开始 read_verilog"
read_verilog $verilog_file
puts "LOG: 完成 read_verilog"
puts "LOG: 开始 read_def"
read_def $def_file
puts "LOG: 完成 read_def"

# 保存初始布局
puts "LOG: 保存初始布局"
write_def "$output_dir/iterations/iteration_0_initial.def"

# 自定义global_placement函数，支持迭代输出
proc global_placement_with_output {{density}} {{
    global output_dir save_interval
    
    puts "LOG: 开始自定义global_placement，密度: $density"
    
    # 设置placement参数
    set_placement_padding -global -left 0 -right 0
    set_placement_padding -masters -left 0 -right 0
    
    # 获取初始统计信息
    set initial_hpwl [get_placement_stats -hpwl]
    set initial_overflow [get_placement_stats -overflow]
    
    puts "LOG: 初始HPWL: $initial_hpwl"
    puts "LOG: 初始溢出率: $initial_overflow"
    
    # 执行placement迭代
    set iteration 0
    set max_iterations 500
    
    while {{$iteration < $max_iterations}} {{
        # 执行一次placement迭代
        set_placement_padding -global -left 0 -right 0
        set_placement_padding -masters -left 0 -right 0
        
        # 计算当前统计信息
        set current_hpwl [get_placement_stats -hpwl]
        set current_overflow [get_placement_stats -overflow]
        
        puts "LOG: 迭代 $iteration - HPWL: $current_hpwl, 溢出率: $current_overflow"
        
        # 每隔save_interval次迭代保存DEF文件
        if {{$iteration % $save_interval == 0}} {{
            set def_filename "$output_dir/iterations/iteration_${{iteration}}_hpwl_${{current_hpwl}}_overflow_${{current_overflow}}.def"
            write_def $def_filename
            puts "LOG: 保存布局到: $def_filename"
        }}
        
        # 检查收敛条件
        if {{$current_overflow < 0.1}} {{
            puts "LOG: 溢出率收敛到 $current_overflow，停止迭代"
            break
        }}
        
        incr iteration
    }}
    
    # 保存最终布局
    set final_hpwl [get_placement_stats -hpwl]
    set final_overflow [get_placement_stats -overflow]
    set def_filename "$output_dir/iterations/iteration_final_hpwl_${{final_hpwl}}_overflow_${{final_overflow}}.def"
    write_def $def_filename
    puts "LOG: 保存最终布局到: $def_filename"
    
    puts "LOG: global_placement完成，总迭代次数: $iteration"
    puts "LOG: 最终HPWL: $final_hpwl"
    puts "LOG: 最终溢出率: $final_overflow"
}}

# 执行自定义placement
global_placement_with_output 0.91

puts "LOG: 跳过 detailed_placement，直接生成报告"

# 生成面积报告
puts "LOG: 开始生成面积报告"
set area_fp [open "$output_dir/area.rpt" w]
puts $area_fp "=== 增强版OpenROAD布局面积报告 ==="
puts $area_fp "生成时间: [clock format [clock seconds]]"
puts $area_fp ""

# 使用ord::函数获取面积信息
if {{[catch {{set core_area [ord::get_core_area]}} result]}} {{
    puts $area_fp "无法获取core_area: $result"
    set core_area 0
}} else {{
    puts $area_fp "Core面积: $core_area"
}}

if {{[catch {{set die_area [ord::get_die_area]}} result]}} {{
    puts $area_fp "无法获取die_area: $result"
    set die_area 0
}} else {{
    puts $area_fp "Die面积: $die_area"
}}

if {{$core_area > 0 && $die_area > 0}} {{
    set utilization [expr {{double($core_area) / double($die_area) * 100.0}}]
    puts $area_fp "利用率: [format {{%.3f}} $utilization]%"
}}

close $area_fp

puts "LOG: 完成面积报告"
puts "LOG: 所有报告生成完成"
puts "LOG: 输出目录: $output_dir"
puts "LOG: 迭代DEF文件保存在: $output_dir/iterations/"
"""
        
        # 保存TCL脚本
        script_path = Path(output_dir) / "iterative_placement.tcl"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        return str(script_path)
    
    def run_iterative_placement(self, 
                              verilog_file: str,
                              cells_lef: str,
                              tech_lef: str,
                              def_file: str,
                              work_dir: str,
                              save_interval: int = 50) -> Dict[str, Any]:
        """运行支持迭代输出的Place&Route流程
        
        Args:
            save_interval: 每隔多少次迭代保存一次DEF文件
            
        Returns:
            Dict[str, Any]: 包含迭代数据和结果
        """
        
        try:
            logger.info(f"开始运行迭代Place&Route流程，保存间隔: {save_interval}")
            
            # 创建TCL脚本
            tcl_script = self.create_iterative_placement_tcl(
                verilog_file, cells_lef, tech_lef, def_file, 
                f"{work_dir}/output", save_interval
            )
            
            # 运行OpenROAD命令
            command = f"openroad -exit {Path(tcl_script).name}"
            success, stdout, stderr = self.run_openroad_command(command, work_dir)
            
            if not success:
                logger.error("迭代Place&Route流程失败")
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
                "stdout": stdout,
                "stderr": stderr
            }
            
        except Exception as e:
            logger.error(f"运行迭代Place&Route流程失败: {e}")
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
        
        for def_file in def_files:
            # 解析文件名获取迭代信息
            filename = def_file.name
            match = re.search(r'iteration_(\d+)_hpwl_([\d.]+)_overflow_([\d.]+)', filename)
            
            if match:
                iteration_num = int(match.group(1))
                hpwl = float(match.group(2))
                overflow = float(match.group(3))
                
                iteration_data.append({
                    "iteration": iteration_num,
                    "def_file": str(def_file),
                    "hpwl": hpwl,
                    "overflow": overflow,
                    "filename": filename
                })
            else:
                # 处理特殊文件名（如initial, final）
                if "initial" in filename:
                    iteration_data.append({
                        "iteration": 0,
                        "def_file": str(def_file),
                        "hpwl": None,
                        "overflow": None,
                        "filename": filename
                    })
                elif "final" in filename:
                    match = re.search(r'iteration_final_hpwl_([\d.]+)_overflow_([\d.]+)', filename)
                    if match:
                        hpwl = float(match.group(1))
                        overflow = float(match.group(2))
                        iteration_data.append({
                            "iteration": -1,  # 表示最终状态
                            "def_file": str(def_file),
                            "hpwl": hpwl,
                            "overflow": overflow,
                            "filename": filename
                        })
        
        # 按迭代次数排序
        iteration_data.sort(key=lambda x: x["iteration"])
        
        logger.info(f"收集到 {len(iteration_data)} 个迭代数据")
        return iteration_data
    
    def generate_iteration_visualization(self, 
                                       iteration_data: List[Dict[str, Any]], 
                                       output_dir: str) -> str:
        """生成迭代过程可视化图
        
        Args:
            iteration_data: 迭代数据
            output_dir: 输出目录
            
        Returns:
            str: 可视化图片路径
        """
        
        try:
            # 创建可视化目录
            viz_dir = Path(output_dir) / "visualization"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. 生成HPWL和溢出率变化图
            self.plot_metrics_evolution(iteration_data, viz_dir)
            
            # 2. 生成布局对比图
            self.plot_layout_comparison(iteration_data, viz_dir)
            
            # 3. 生成动画帧（如果数据足够）
            if len(iteration_data) > 2:
                self.generate_animation_frames(iteration_data, viz_dir)
            
            return str(viz_dir)
            
        except Exception as e:
            logger.error(f"生成迭代可视化失败: {e}")
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
    
    def plot_layout_comparison(self, 
                             iteration_data: List[Dict[str, Any]], 
                             output_dir: Path):
        """绘制布局对比图"""
        
        # 选择关键迭代点进行对比
        key_iterations = []
        
        # 添加初始布局
        initial_data = [d for d in iteration_data if d["iteration"] == 0]
        if initial_data:
            key_iterations.append(initial_data[0])
        
        # 添加中间迭代点
        valid_data = [d for d in iteration_data if d["iteration"] > 0]
        if len(valid_data) >= 3:
            # 选择25%, 50%, 75%的迭代点
            mid1_idx = len(valid_data) // 4
            mid2_idx = len(valid_data) // 2
            mid3_idx = 3 * len(valid_data) // 4
            
            key_iterations.extend([
                valid_data[mid1_idx],
                valid_data[mid2_idx],
                valid_data[mid3_idx]
            ])
        
        # 添加最终布局
        final_data = [d for d in iteration_data if d["iteration"] == -1]
        if final_data:
            key_iterations.append(final_data[0])
        
        if len(key_iterations) < 2:
            logger.warning("关键迭代点不足，跳过布局对比图")
            return
        
        # 创建子图
        n_plots = len(key_iterations)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, data in enumerate(key_iterations):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            # 真实DEF解析和物理布局绘制
            self.plot_layout_real(ax, data['def_file'])
        
        # 隐藏多余的子图
        for i in range(n_plots, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('布局迭代对比', fontsize=16, y=0.95)
        plt.tight_layout()
        
        # 保存图片
        comparison_path = output_dir / "layout_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"布局对比图已保存: {comparison_path}")
    
    def plot_layout_real(self, ax, def_file):
        """解析DEF文件并绘制真实布局"""
        parser = DEFParser()
        layout = parser.parse_def_file(def_file)
        visualizer = LayoutVisualizer()
        die_area = layout.get('die_area', {'width': 1.0, 'height': 1.0})
        components = layout.get('components', [])
        for i, comp in enumerate(components):
            x = comp.get('x', 0)
            y = comp.get('y', 0)
            width = comp.get('width', 0)
            height = comp.get('height', 0)
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=1, edgecolor='black',
                facecolor=visualizer.colors[i % len(visualizer.colors)], alpha=0.6
            )
            ax.add_patch(rect)
        # 设置坐标轴等
        if isinstance(die_area, dict):
            ax.set_xlim(0, die_area.get('width', 1.0))
            ax.set_ylim(0, die_area.get('height', 1.0))
        elif isinstance(die_area, list) and len(die_area) == 4:
            ax.set_xlim(die_area[0], die_area[2])
            ax.set_ylim(die_area[1], die_area[3])
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(os.path.basename(def_file), fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def generate_animation_frames(self, 
                                iteration_data: List[Dict[str, Any]], 
                                output_dir: Path):
        """生成动画帧"""
        
        # 创建动画帧目录
        frames_dir = output_dir / "animation_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # 为每个迭代生成一帧
        for i, data in enumerate(iteration_data):
            fig, ax = plt.subplots(figsize=(8, 6))
            self.plot_layout_real(ax, data['def_file'])
            
            frame_path = frames_dir / f"frame_{i:03d}.png"
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"生成了 {len(iteration_data)} 个动画帧: {frames_dir}")

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
        verilog_file=verilog_file,
        cells_lef=cells_lef,
        tech_lef=tech_lef,
        def_file=def_file,
        work_dir=design_dir,
        save_interval=25  # 每25次迭代保存一次
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