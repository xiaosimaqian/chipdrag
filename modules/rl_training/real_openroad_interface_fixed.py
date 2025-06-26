#!/usr/bin/env python3
"""
OpenROAD接口模块 - 用于强化学习训练 (修正版)
根据OpenROAD官方手册修正，提供与OpenROAD工具的交互接口，支持布局优化和评估
"""

import os
import subprocess
import tempfile
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealOpenROADInterface:
    """OpenROAD真实接口类 (修正版)"""
    
    def __init__(self, 
                 work_dir: str = "/Users/keqin/Documents/workspace/chip-rag/chipdrag/data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1",
                 use_lib: bool = False):
        """
        初始化OpenROAD接口
        
        Args:
            work_dir: 工作目录路径
            use_lib: 是否使用LIB文件
        """
        self.work_dir = Path(work_dir)
        self.use_lib = use_lib
        
        # 文件路径配置
        self.verilog_file = self.work_dir / "design.v"
        self.def_file = self.work_dir / "floorplan.def"  # 使用初始无布局的DEF文件
        self.tech_lef = self.work_dir / "tech.lef"
        self.cells_lef = self.work_dir / "cells.lef"
        self.lib_files = list(self.work_dir.glob("*.lib"))
        
        # 验证文件存在性
        self._validate_files()
        
        logger.info(f"OpenROAD接口初始化完成，工作目录: {self.work_dir}")
        logger.info(f"Verilog文件: {self.verilog_file}")
        logger.info(f"DEF文件: {self.def_file}")
        logger.info(f"LEF文件: {self.tech_lef}, {self.cells_lef}")
        logger.info(f"LIB文件数量: {len(self.lib_files)}")
    
    def _validate_files(self):
        """验证必要文件是否存在"""
        required_files = [
            self.verilog_file,
            self.def_file,
            self.tech_lef,
            self.cells_lef
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            raise FileNotFoundError(f"缺少必要文件: {missing_files}")
    
    def _check_lib_files(self) -> bool:
        """检查LIB文件是否存在"""
        has_lib_files = len(self.lib_files) > 0
        if has_lib_files:
            logger.info(f"发现LIB文件: {[f.name for f in self.lib_files]}")
        else:
            logger.info("未发现LIB文件")
        return has_lib_files
    
    def _detect_top_module(self) -> str:
        """检测Verilog文件中的顶层模块名"""
        try:
            with open(self.verilog_file, 'r') as f:
                content = f.read()
            
            # 使用正则表达式查找module声明
            import re
            module_pattern = r'module\s+(\w+)\s*\('
            match = re.search(module_pattern, content)
            
            if match:
                top_module = match.group(1)
                logger.info(f"检测到顶层模块: {top_module}")
                return top_module
            else:
                logger.warning("无法检测到顶层模块，使用默认名称: des_perf")
                return "des_perf"
        except Exception as e:
            logger.error(f"检测顶层模块失败: {e}")
            return "des_perf"
    
    def _generate_tcl_script(self, 
                            density_target: float = 0.7,
                            wirelength_weight: float = 1.0,
                            density_weight: float = 1.0) -> str:
        """
        生成OpenROAD TCL脚本 (根据官方API文档修正)
        
        Args:
            density_target: 密度目标
            wirelength_weight: 线长权重
            density_weight: 密度权重
            
        Returns:
            TCL脚本内容
        """
        # 检查LIB文件
        has_lib_files = self._check_lib_files()
        
        # 根据OpenROAD手册，先读取技术LEF，再读取单元LEF
        lef_read_cmds = [
            f"read_lef -tech {self.tech_lef.name}",
            f"read_lef -library {self.cells_lef.name}"
        ]
        lef_read_cmds_str = "\n".join(lef_read_cmds)
        
        # 生成LIB读取命令
        lib_read_cmds = []
        if self.use_lib and has_lib_files:
            for lib_file in self.lib_files:
                lib_read_cmds.append(f"read_liberty {lib_file.name}")
        lib_read_cmds_str = "\n".join(lib_read_cmds)
        
        # 根据是否使用LIB且有LIB文件生成不同的TCL脚本
        if self.use_lib and has_lib_files:
            # 有LIB文件且启用LIB模式
            tcl_script = f"""
# OpenROAD布局优化脚本 (LIB模式 - 有LIB文件)
# 设计: {self.verilog_file.name}
# 参数: density_target={density_target}, wirelength_weight={wirelength_weight}, density_weight={density_weight}

# 完全重置数据库 - 确保每次运行都是全新状态
if {{[info exists ::ord::db]}} {{
    ord::reset_db
}}

# 读取LEF文件 - 按照OpenROAD手册顺序
{lef_read_cmds_str}

# 读取LIB文件
{lib_read_cmds_str}

# 读取Verilog文件
read_verilog {self.verilog_file.name}

# 连接设计 - 指定顶层模块名
link_design {self._detect_top_module()}

# 初始化布局 - 使用默认站点
initialize_floorplan -die_area "0 0 1000 1000" -core_area "10 10 990 990" -site core

# 设置布局参数
set_placement_padding -global -left 2 -right 2

# 全局布局
global_placement -density {density_target}

# 详细布局
detailed_placement

# 输出结果
write_def -output placement_result.def
write_verilog -output placement_result.v

# 报告结果
report_placement
report_timing
report_area

puts "OpenROAD布局优化完成"
"""
        else:
            # 无LIB文件或未启用LIB模式
            tcl_script = f"""
# OpenROAD布局优化脚本 (无LIB模式)
# 设计: {self.verilog_file.name}
# 参数: density_target={density_target}, wirelength_weight={wirelength_weight}, density_weight={density_weight}

# 完全重置数据库 - 确保每次运行都是全新状态
if {{[info exists ::ord::db]}} {{
    ord::reset_db
}}

# 读取LEF文件 - 按照OpenROAD手册顺序
{lef_read_cmds_str}

# 读取Verilog文件
read_verilog {self.verilog_file.name}

# 连接设计 - 指定顶层模块名
link_design {self._detect_top_module()}

# 初始化布局 - 使用默认站点
initialize_floorplan -die_area "0 0 1000 1000" -core_area "10 10 990 990" -site core

# 设置布局参数
set_placement_padding -global -left 2 -right 2

# 全局布局
global_placement -density {density_target}

# 详细布局
detailed_placement

# 输出结果
write_def -output placement_result.def
write_verilog -output placement_result.v

# 报告结果
report_placement
report_timing
report_area

puts "OpenROAD布局优化完成"
"""
        
        return tcl_script
    
    def run_placement(self, 
                     density_target: float = 0.7,
                     wirelength_weight: float = 1.0,
                     density_weight: float = 1.0) -> Dict[str, Any]:
        """
        运行OpenROAD布局优化
        
        Args:
            density_target: 密度目标
            wirelength_weight: 线长权重
            density_weight: 密度权重
            
        Returns:
            执行结果字典
        """
        try:
            # 生成TCL脚本
            tcl_script = self._generate_tcl_script(
                density_target=density_target,
                wirelength_weight=wirelength_weight,
                density_weight=density_weight
            )
            
            # 写入TCL文件
            tcl_file = self.work_dir / "openroad_script.tcl"
            with open(tcl_file, 'w') as f:
                f.write(tcl_script)
            
            logger.info(f"TCL脚本已生成: {tcl_file}")
            
            # 确保工作目录是绝对路径
            work_dir_abs = self.work_dir.resolve()
            
            # 构建Docker命令 - 使用直接执行方式
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{work_dir_abs}:/workspace",
                "-w", "/workspace",
                "openroad/flow-ubuntu22.04-builder:21e414",
                "bash", "-c",
                f"export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH && openroad -no_init -no_splash -exit openroad_script.tcl"
            ]
            
            logger.info(f"执行Docker命令: {' '.join(docker_cmd)}")
            
            # 直接执行Docker命令
            import subprocess, time
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    docker_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=1800  # 30分钟超时
                )
                execution_time = time.time() - start_time
                
                # 处理输出
                stdout_lines = result.stdout.split('\n') if result.stdout else []
                stderr_lines = result.stderr.split('\n') if result.stderr else []
                
                # 检查是否成功
                success = result.returncode == 0 and any("OpenROAD布局优化完成" in l for l in stdout_lines)
                
                # 提取关键指标
                metrics = self._extract_metrics(stdout_lines, stderr_lines)
                
                return {
                    'success': success,
                    'return_code': result.returncode,
                    'execution_time': execution_time,
                    'stdout': stdout_lines,
                    'stderr': stderr_lines,
                    'metrics': metrics
                }
                
            except subprocess.TimeoutExpired:
                logger.error("OpenROAD执行超时")
                return {
                    'success': False,
                    'return_code': -1,
                    'execution_time': time.time() - start_time,
                    'stdout': [],
                    'stderr': ["执行超时"],
                    'metrics': {}
                }
                
        except Exception as e:
            logger.error(f"运行OpenROAD时发生错误: {e}")
            return {
                'success': False,
                'return_code': -1,
                'execution_time': 0,
                'stdout': [],
                'stderr': [str(e)],
                'metrics': {}
            }
    
    def _analyze_output(self, stdout: List[str], stderr: List[str]) -> Dict[str, Any]:
        """
        分析OpenROAD输出
        
        Args:
            stdout: 标准输出
            stderr: 标准错误
            
        Returns:
            分析结果
        """
        analysis = {
            "errors": [],
            "warnings": [],
            "info_messages": [],
            "placement_stats": {}
        }
        
        # 分析错误
        for line in stderr:
            if '[ERROR' in line:
                analysis["errors"].append(line.strip())
            elif '[WARNING' in line:
                analysis["warnings"].append(line.strip())
            elif '[INFO' in line:
                analysis["info_messages"].append(line.strip())
        
        # 分析标准输出
        for line in stdout:
            if '[INFO' in line:
                analysis["info_messages"].append(line.strip())
            elif 'placement' in line.lower() and 'complete' in line.lower():
                analysis["placement_stats"]["placement_completed"] = True
        
        return analysis
    
    def get_placement_quality(self, result_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        获取布局质量指标
        
        Args:
            result_dict: 执行结果字典
            
        Returns:
            质量指标字典
        """
        if not result_dict.get("success", False):
            return {
                "wirelength": float('inf'),
                "density": 0.0,
                "congestion": float('inf'),
                "timing": float('inf')
            }
        
        # 这里可以添加更详细的质量分析逻辑
        # 目前返回基本指标
        return {
            "wirelength": 1000.0,  # 示例值
            "density": 0.7,         # 示例值
            "congestion": 0.1,      # 示例值
            "timing": 1.0           # 示例值
        }

    def _extract_metrics(self, stdout_lines: List[str], stderr_lines: List[str]) -> Dict[str, Any]:
        """
        从OpenROAD输出中提取关键指标
        
        Args:
            stdout_lines: 标准输出行列表
            stderr_lines: 错误输出行列表
            
        Returns:
            提取的指标字典
        """
        metrics = {
            'wirelength': None,
            'area': None,
            'density': None,
            'timing': None,
            'overflow': None
        }
        
        # 合并所有输出行进行分析
        all_lines = stdout_lines + stderr_lines
        
        for line in all_lines:
            line = line.strip()
            
            # 提取线长信息
            if 'wirelength' in line.lower():
                try:
                    # 查找数字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        metrics['wirelength'] = float(numbers[0])
                except:
                    pass
            
            # 提取面积信息
            elif 'area' in line.lower() and 'total' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        metrics['area'] = float(numbers[0])
                except:
                    pass
            
            # 提取密度信息
            elif 'density' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        metrics['density'] = float(numbers[0])
                except:
                    pass
            
            # 提取时序信息
            elif 'slack' in line.lower() or 'timing' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if numbers:
                        metrics['timing'] = float(numbers[0])
                except:
                    pass
            
            # 提取溢出信息
            elif 'overflow' in line.lower():
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        metrics['overflow'] = float(numbers[0])
                except:
                    pass
        
        return metrics

def main():
    """主函数 - 测试接口"""
    try:
        # 创建接口实例
        interface = RealOpenROADInterface()
        
        # 运行布局优化
        result = interface.run_placement(
            density_target=0.7,
            wirelength_weight=1.0,
            density_weight=1.0
        )
        
        # 输出结果
        print("=== 布局优化结果 ===")
        print(f"成功: {result['success']}")
        print(f"返回码: {result['return_code']}")
        print(f"执行时间: {result['execution_time']:.2f}秒")
        
        if result['success']:
            print("=== 标准输出 ===")
            print("\n".join(result['stdout']))
            
            print("=== 输出分析 ===")
            analysis = result['output_analysis']
            print(f"错误数量: {len(analysis['errors'])}")
            print(f"警告数量: {len(analysis['warnings'])}")
            print(f"信息数量: {len(analysis['info_messages'])}")
            
            if analysis['errors']:
                print("=== 错误信息 ===")
                for error in analysis['errors']:
                    print(error)
        else:
            print("=== 错误信息 ===")
            print("\n".join(result['stderr']))
        
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main() 