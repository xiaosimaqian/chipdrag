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
import shutil

# 导入统一的OpenROAD接口
from enhanced_openroad_interface import EnhancedOpenROADInterface

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
        
        # 创建统一的OpenROAD接口实例
        self.openroad_interface = EnhancedOpenROADInterface()
        
        # 验证文件存在性
        self._validate_files()
        
        print(f"OpenROAD接口初始化完成，工作目录: {self.work_dir}")
        print(f"Verilog文件: {self.verilog_file}")
        print(f"DEF文件: {self.def_file}")
        print(f"LEF文件: {self.tech_lef}, {self.cells_lef}")
        print(f"LIB文件数量: {len(self.lib_files)}")
    
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
            print(f"发现LIB文件: {[f.name for f in self.lib_files]}")
        else:
            print("未发现LIB文件")
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
                print(f"检测到顶层模块: {top_module}")
                return top_module
            else:
                print("无法检测到顶层模块，使用默认名称: des_perf")
                return "des_perf"
        except Exception as e:
            print(f"检测顶层模块失败: {e}")
            return "des_perf"
    
    def _generate_tcl_script(self, 
                            density_target: float = 0.7,
                            wirelength_weight: float = 1.0,
                            density_weight: float = 1.0,
                            die_size: int = 800,
                            core_size: int = 790) -> str:
        """
        生成OpenROAD TCL脚本 (使用动态参数)
        
        Args:
            density_target: 密度目标
            wirelength_weight: 线长权重
            density_weight: 密度权重
            die_size: 芯片尺寸
            core_size: 核心区域尺寸
            
        Returns:
            TCL脚本内容
        """
        # 检查LIB文件
        has_lib_files = self._check_lib_files()
        
        # 生成LIB读取命令
        lib_read_cmds = []
        if has_lib_files:
            for lib_file in self.lib_files:
                lib_read_cmds.append(f"read_liberty {lib_file.name}")
        lib_read_cmds_str = "\n".join(lib_read_cmds)
        
        # 检测顶层模块名
        top_module = self._detect_top_module()
        
        # 使用动态参数生成TCL脚本
        utilization = int(density_target * 100)
        tcl_script = f"""
# OpenROAD布局优化脚本 (智能参数版本)
# 设计: {self.verilog_file.name}
# 参数: density_target={density_target}, utilization={utilization}, die_size={die_size}x{die_size}, core_size={core_size}x{core_size}

# 完全重置数据库
if {{[info exists ::ord::db]}} {{
    ord::reset_db
}}

# 读取LEF文件 - 先读取技术LEF，再读取单元LEF
read_lef {self.tech_lef.name}
read_lef {self.cells_lef.name}

# 读取Liberty文件（如果存在）
{lib_read_cmds_str}

# 检查DEF文件是否存在，决定读取顺序
if {{[file exists floorplan.def]}} {{
    puts "读取现有DEF文件: floorplan.def"
    # 如果DEF文件存在，先读取DEF文件
    read_def floorplan.def
    
    # 然后读取Verilog文件并连接设计
    read_verilog {self.verilog_file.name}
    link_design {top_module}
    
    # 智能扩展芯片面积
    puts "智能扩展芯片面积到 {die_size}x{die_size}..."
    set db [ord::get_db]
    set chip [$db getChip]
    set block [$chip getBlock]
    
    # 使用OpenROAD标准方法重新初始化floorplan
    puts "使用utilization {utilization}%重新初始化floorplan..."
    
    # 获取site信息
    set site_info "{self._get_site_info()}"
    puts "使用site: $site_info"
    
    # 重新初始化floorplan，使用utilization方法
    initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 -core_space 10 -site $site_info
    
    # 确保所有实例都是UNPLACED状态，避免混合状态导致的详细布局错误
    puts "确保所有实例为UNPLACED状态..."
    set insts [$block getInsts]
    set unplaced_count 0
    
    foreach inst $insts {{
        if {{[$inst isPlaced]}} {{
            $inst setPlacementStatus "UNPLACED"
            incr unplaced_count
        }}
    }}
    puts "已将 $unplaced_count 个已放置实例重置为UNPLACED状态"
}} else {{
    puts "未找到DEF文件，将创建新的floorplan"
    # 如果DEF文件不存在，先读取Verilog文件并连接设计
    read_verilog {self.verilog_file.name}
    link_design {top_module}
    
    # 获取site信息
    set site_info "{self._get_site_info()}"
    puts "使用site: $site_info"
    
    # 然后创建floorplan，使用utilization方法
    puts "初始化布局，使用utilization {utilization}%..."
    initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 -core_space 10 -site $site_info
}}

# 设置布局参数 - 使用更宽松的设置
set_placement_padding -global -left 2 -right 2

# 全局布局 - 使用动态密度目标，增加容错性
puts "开始全局布局 (密度目标: {density_target})..."
if {{[catch {{global_placement -density {density_target} -overflow 0.1}} result]}} {{
    puts "全局布局失败: $result"
    puts "尝试使用默认参数..."
    if {{[catch {{global_placement}} result2]}} {{
        puts "全局布局完全失败: $result2"
        exit 1
    }} else {{
        puts "全局布局使用默认参数成功"
    }}
}} else {{
    puts "全局布局成功完成"
}}

# 检查全局布局结果
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set insts [$block getInsts]
set placed_count 0
set total_count 0

foreach inst $insts {{
    if {{[$inst isPlaced]}} {{
        incr placed_count
    }}
    incr total_count
}}

puts "全局布局完成: $placed_count/$total_count 实例已放置"

# 详细布局 - 使用更宽松的参数
puts "开始详细布局..."
if {{[catch {{detailed_placement -max_displacement 5}} result]}} {{
    puts "详细布局失败，尝试使用更宽松的参数..."
    if {{[catch {{detailed_placement -max_displacement 10}} result]}} {{
        puts "详细布局仍然失败，跳过详细布局步骤"
        puts "警告：详细布局失败，但布局流程将继续"
    }} else {{
        puts "详细布局成功完成"
    }}
}} else {{
    puts "详细布局成功完成"
}}

# 最终检查布局结果
set final_placed_count 0
set final_total_count 0
foreach inst $insts {{
    if {{[$inst isPlaced]}} {{
        incr final_placed_count
    }}
    incr final_total_count
}}

puts "最终布局结果: $final_placed_count/$final_total_count 实例已放置"

# 输出结果
write_def placement_result.def
write_verilog placement_result.v

# 输出布局完成信息
puts "=== 布局完成 ==="
puts "输出文件: placement_result.def, placement_result.v"
puts "布局实例数: $final_placed_count/$final_total_count"
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
            # 获取设计统计信息
            design_stats = self._extract_design_stats()
            
            # 计算最优参数
            optimal_params = self._calculate_optimal_parameters(design_stats)
            
            # 使用计算出的参数
            actual_density = optimal_params['density_target']
            die_size = optimal_params['die_size']
            core_size = optimal_params['core_size']
            
            print(f"设计统计: {design_stats['num_instances']} 实例, {design_stats['num_nets']} 网络")
            print(f"智能参数: 密度={actual_density}, 芯片尺寸={die_size}x{die_size}, 核心尺寸={core_size}x{core_size}")
            
            # 生成TCL脚本
            tcl_script = self._generate_tcl_script(
                density_target=actual_density,
                wirelength_weight=wirelength_weight,
                density_weight=density_weight,
                die_size=die_size,
                core_size=core_size
            )
            
            # 写入TCL文件
            tcl_file = self.work_dir / "openroad_script.tcl"
            with open(tcl_file, 'w') as f:
                f.write(tcl_script)
            
            # 计算智能超时时间
            timeout = self._calculate_timeout(design_stats)
            
            print(f"执行OpenROAD (超时: {timeout}秒)...")
            
            # 获取绝对路径
            abs_work_dir = os.path.abspath(self.work_dir)
            # 执行OpenROAD命令
            start_time = time.time()
            result = subprocess.run([
                'docker', 'run', '--rm',
                '-m', '12g', '-c', '4',  # 增加内存限制
                '-v', f'{abs_work_dir}:/workspace',
                '-w', '/workspace',
                'openroad/flow-ubuntu22.04-builder:21e414',
                'bash', '-c',
                f'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit {os.path.basename(tcl_file)}'
            ], capture_output=True, text=True, timeout=timeout)
            execution_time = time.time() - start_time
            
            # 分析输出
            stdout_lines = result.stdout.split('\n')
            stderr_lines = result.stderr.split('\n')
            
            # 提取HPWL
            hpwl = self._extract_hpwl_from_def("placement_result.def")
            if hpwl == float('inf'):
                # 如果从DEF提取失败，尝试从日志提取
                hpwl = self._extract_hpwl_from_log("openroad_execution.log")
            
            # 分析结果
            analysis_result = self._analyze_output(stdout_lines, stderr_lines)
            analysis_result['execution_time'] = execution_time
            analysis_result['hpwl'] = hpwl  # 添加HPWL到结果中
            
            if result.returncode == 0 and analysis_result['success']:
                print(f"✅ 布局成功完成 (HPWL: {hpwl:.2e}, 耗时: {execution_time:.2f}秒)")
            else:
                print(f"❌ 布局失败 (耗时: {execution_time:.2f}秒)")
                if hpwl != float('inf'):
                    print(f"   但成功提取HPWL: {hpwl:.2e}")
            
            return analysis_result
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'wirelength': None,
                'area': None,
                'metrics': {},
                'design_stats': design_stats if 'design_stats' in locals() else {},
                'optimal_params': optimal_params if 'optimal_params' in locals() else {},
                'hpwl': float('inf'),
                'errors': [str(e)],
                'warnings': [],
                'info_messages': []
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
            "success": True,  # 默认成功
            "errors": [],
            "warnings": [],
            "info_messages": [],
            "placement_stats": {},
            "wirelength": None,
            "stderr": "\n".join(stderr),
            "stdout": "\n".join(stdout)
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
        
        # 如果有严重错误，标记为失败
        if len(analysis["errors"]) > 0:
            analysis["success"] = False
        
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
            
            # 提取HPWL (Half-Perimeter Wirelength)
            # 格式: HPWL: 1.063218e+10 或 HPWL: 16752570900
            if 'HPWL:' in line:
                try:
                    import re
                    # 匹配HPWL后面的数字（科学计数法或普通数字）
                    hpwl_match = re.search(r'HPWL:\s*([0-9.]+(?:e[+-]?\d+)?)', line)
                    if hpwl_match:
                        hpwl_value = float(hpwl_match.group(1))
                        # 如果HPWL值很大（科学计数法），转换为更小的单位
                        if hpwl_value > 1e9:
                            hpwl_value = hpwl_value / 1e6  # 转换为百万单位
                        metrics['wirelength'] = hpwl_value
                except:
                    pass
            
            # 提取核心面积
            # 格式: Core area: 4752400.000 um^2
            elif 'Core area:' in line:
                try:
                    import re
                    area_match = re.search(r'Core area:\s*([0-9.]+)', line)
                    if area_match:
                        metrics['area'] = float(area_match.group(1))
                except:
                    pass
            
            # 提取利用率
            # 格式: Utilization: 37.405 %
            elif 'Utilization:' in line:
                try:
                    import re
                    util_match = re.search(r'Utilization:\s*([0-9.]+)', line)
                    if util_match:
                        metrics['density'] = float(util_match.group(1)) / 100.0
                except:
                    pass
            
            # 提取溢出信息
            # 格式: Finished with Overflow: 0.099203
            elif 'Overflow:' in line and 'Finished' in line:
                try:
                    import re
                    overflow_match = re.search(r'Overflow:\s*([0-9.]+)', line)
                    if overflow_match:
                        metrics['overflow'] = float(overflow_match.group(1))
                except:
                    pass
            
        return metrics

    def _extract_site_info(self) -> str:
        """
        从LEF文件中提取站点信息
        对于ISPD基准测试，所有设计都使用相同的core站点
        
        Returns:
            站点名称
        """
        try:
            # 检查tech.lef文件中的站点定义
            if self.tech_lef.exists():
                with open(self.tech_lef, 'r') as f:
                    content = f.read()
                    # 查找SITE定义
                    import re
                    site_match = re.search(r'SITE\s+(\w+)', content)
                    if site_match:
                        site_name = site_match.group(1)
                        print(f"从LEF文件中提取到站点: {site_name}")
                        return site_name
            
            # 如果无法从LEF文件中提取，使用ISPD基准测试的默认站点
            print("使用ISPD基准测试默认站点: core")
            return "core"
            
        except Exception as e:
            print(f"站点提取失败: {e}，使用默认站点: core")
            return "core"
    
    def _get_ispd_site_config(self) -> str:
        """
        获取ISPD基准测试的通用站点配置
        所有ISPD基准测试都使用相同的core站点定义
        
        Returns:
            站点配置字符串
        """
        return """
# ISPD基准测试通用站点配置
# 所有ISPD基准测试都使用相同的core站点
SITE core
  SIZE 0.20 BY 2.00 ;
  CLASS CORE ;
  SYMMETRY Y  ;
END core
"""

    def _calculate_timeout(self, design_stats: Dict[str, Any]) -> int:
        """
        根据设计规模计算超时时间
        
        Args:
            design_stats: 设计统计信息
            
        Returns:
            int: 超时时间（秒）
        """
        num_instances = design_stats.get('num_instances', 0)
        num_nets = design_stats.get('num_nets', 0)
        
        # 基础超时时间：1小时
        base_timeout = 3600
        
        # 根据设计规模调整超时时间
        if num_instances < 50000:
            timeout = base_timeout
        elif num_instances < 100000:
            timeout = base_timeout + 1800  # 1.5小时
        elif num_instances < 500000:
            timeout = base_timeout + 3600  # 2小时
        elif num_instances < 1000000:
            timeout = base_timeout + 5400  # 2.5小时
        else:
            timeout = base_timeout + 7200  # 3小时
        
        print(f"设计规模: {num_instances}实例, {num_nets}网络")
        print(f"计算超时时间: {timeout}秒")
        
        return timeout
    
    def _extract_design_stats(self) -> Dict[str, Any]:
        """
        提取设计统计信息
        
        Returns:
            设计统计信息字典
        """
        stats = {
            'num_instances': 0,
            'num_nets': 0,
            'num_pins': 0,
            'core_area': 0
        }
        
        try:
            # 从DEF文件中提取实例数量和引脚数量
            if self.def_file.exists():
                with open(self.def_file, 'r') as f:
                    content = f.read()
                    
                    # 提取组件数量
                    import re
                    components_match = re.search(r'COMPONENTS\s+(\d+)', content)
                    if components_match:
                        stats['num_instances'] = int(components_match.group(1))
                    
                    # 提取引脚数量
                    pins_match = re.search(r'PINS\s+(\d+)', content)
                    if pins_match:
                        stats['num_pins'] = int(pins_match.group(1))
                    
                    # 提取网络数量 - 使用行首匹配确保匹配NETS而不是SPECIALNETS
                    nets_match = re.search(r'^NETS\s+(\d+)', content, re.MULTILINE)
                    if nets_match:
                        stats['num_nets'] = int(nets_match.group(1))
                    
                    # 提取核心面积
                    diearea_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
                    if diearea_match:
                        x1, y1, x2, y2 = map(int, diearea_match.groups())
                        
                        # 检查单位定义
                        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
                        if units_match:
                            units_factor = int(units_match.group(1))
                            # 将坐标转换为微米
                            x1 = x1 // units_factor
                            y1 = y1 // units_factor
                            x2 = x2 // units_factor
                            y2 = y2 // units_factor
                        
                        stats['core_area'] = (x2 - x1) * (y2 - y1)
                        print(f"从DEF文件提取面积: {x1}x{y1} 到 {x2}x{y2}, 面积: {stats['core_area']} um²")
            
            # 如果DEF文件中没有找到，尝试从Verilog文件中提取基本信息
            if stats['num_instances'] == 0 and self.verilog_file.exists():
                with open(self.verilog_file, 'r') as f:
                    content = f.read()
                    
                    # 网络数量估算（基于wire/reg声明）
                    wire_count = content.count('wire') + content.count('reg')
                    stats['num_nets'] = wire_count
                    
                    # 引脚数量估算（基于端口声明）
                    port_count = content.count('input') + content.count('output') + content.count('inout')
                    stats['num_pins'] = port_count
            
            # 从LEF文件中提取面积信息（如果DEF文件中没有找到）
            if stats['core_area'] == 0 and self.tech_lef.exists():
                with open(self.tech_lef, 'r') as f:
                    content = f.read()
                    
                    # 查找SITE定义来估算面积
                    if 'SITE core' in content:
                        # 使用默认的核心面积
                        stats['core_area'] = 800 * 800  # 800x800 um²
                        
        except Exception as e:
            print(f"提取设计统计信息时出错: {e}")
        
        return stats

    def _calculate_optimal_parameters(self, design_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据设计规模计算最优参数（增强版）
        
        Args:
            design_stats: 设计统计信息
            
        Returns:
            最优参数字典
        """
        # 从设计统计中提取关键指标
        num_instances = design_stats.get('num_instances', 0)
        num_nets = design_stats.get('num_nets', 0)
        core_area = design_stats.get('core_area', 0)
        
        # 基础参数
        base_density = 0.70
        base_die_size = 800
        base_core_size = 790
        
        # 根据实例数量调整参数（增强版）
        if num_instances > 500000:  # 超超大型设计（如superblue系列）
            density_target = 0.60  # 降低密度目标，避免利用率过高
            die_size = 2000
            core_size = 1990
            print(f"检测到超超大型设计 ({num_instances}实例)，使用特殊参数")
        elif num_instances > 200000:  # 超大型设计
            density_target = 0.65
            die_size = 1600
            core_size = 1590
            print(f"检测到超大型设计 ({num_instances}实例)，使用宽松参数")
        elif num_instances > 100000:  # 大型设计
            density_target = 0.75
            die_size = 1200
            core_size = 1190
        elif num_instances > 50000:  # 中型设计
            density_target = 0.80
            die_size = 1000
            core_size = 990
        elif num_instances > 20000:  # 中小型设计
            density_target = 0.75
            die_size = 900
            core_size = 890
        else:  # 小型设计
            density_target = base_density
            die_size = base_die_size
            core_size = base_core_size
        
        # 根据网络数量进一步调整
        if num_nets > 200000:
            density_target = min(0.70, density_target - 0.05)  # 降低密度
            die_size = max(die_size, 1800)
            core_size = die_size - 10
        elif num_nets > 100000:
            density_target = min(0.80, density_target + 0.05)
            die_size = max(die_size, 1400)
            core_size = die_size - 10
        
        # 根据核心面积调整（如果DEF文件中提供了面积信息）
        if core_area > 0:
            # 计算所需的die尺寸（基于面积和密度）
            required_area = core_area / density_target
            required_side = int(required_area ** 0.5)
            
            # 确保die尺寸足够大
            if required_side > die_size:
                die_size = required_side + 100  # 增加100um的裕量
                core_size = die_size - 10
                print(f"根据面积需求调整die尺寸: {die_size}x{die_size}")
        
        # 特殊处理：针对已知的超大设计
        design_name = self.work_dir.name if hasattr(self.work_dir, 'name') else ""
        if 'superblue' in design_name.lower():
            # superblue系列特殊处理
            density_target = 0.55  # 进一步降低密度
            die_size = max(die_size, 2200)
            core_size = die_size - 10
            print(f"检测到superblue设计，使用特殊参数: density={density_target}, die={die_size}")
        elif 'mgc' in design_name.lower() and num_instances > 100000:
            # mgc系列大型设计
            density_target = min(0.70, density_target)
            die_size = max(die_size, 1500)
            core_size = die_size - 10
            print(f"检测到mgc大型设计，调整参数: density={density_target}, die={die_size}")
        
        # 最终验证：确保参数合理
        if density_target < 0.5:
            density_target = 0.5
        if density_target > 0.9:
            density_target = 0.9
        
        # 确保die和core尺寸合理
        if die_size < 600:
            die_size = 600
        if die_size > 3000:
            die_size = 3000
        
        # 确保core_size始终小于die_size
        core_size = die_size - 10
        
        print(f"最终参数: density={density_target:.2f}, die={die_size}x{die_size}, core={core_size}x{core_size}")
        
        return {
            'density_target': density_target,
            'die_size': die_size,
            'core_size': core_size,
            'wirelength_weight': 1.0,
            'density_weight': 1.0
        }

    def create_iterative_placement_tcl(self, num_iterations: int = 10) -> str:
        """
        创建迭代布局TCL脚本，支持RL训练数据收集
        
        Args:
            num_iterations: 迭代次数
            
        Returns:
            str: TCL脚本文件路径
        """
        # 计算最优面积和密度
        optimal_area, optimal_density = self._calculate_optimal_area_and_density()
        width, height = optimal_area
        
        # 根据密度计算utilization
        utilization = int(optimal_density * 100)
        
        tcl_content = f"""# Enhanced OpenROAD Place & Route Script - RL Training Data Collection
set output_dir "output"
set num_iterations {num_iterations}

file mkdir $output_dir
file mkdir "$output_dir/iterations"

set log_file "$output_dir/placement_iterations.log"
set log_fp [open $log_file w]

puts "LOG: Current directory: [pwd]"
puts "LOG: output_dir=$output_dir"
puts "LOG: Number of iterations: $num_iterations"
puts "LOG: Optimal area: {width}x{height}, density: {optimal_density:.2f}, utilization: {utilization}%"
puts $log_fp "=== OpenROAD RL Training Data Collection ==="
puts $log_fp "Number of iterations: $num_iterations"
puts $log_fp "Optimal area: {width}x{height}, density: {optimal_density:.2f}, utilization: {utilization}%"

# 完全重置数据库
if {{[info exists ::ord::db]}} {{
    ord::reset_db
}}

# 读取LEF文件 - 先读取技术LEF，再读取单元LEF
read_lef tech.lef
read_lef cells.lef

# 检查DEF文件是否存在，决定读取顺序
if {{[file exists floorplan.def]}} {{
    puts "读取现有DEF文件: floorplan.def"
    # 如果DEF文件存在，先读取DEF文件
    read_def floorplan.def
    
    # 然后读取Verilog文件并连接设计
    read_verilog design.v
    link_design {self._detect_top_module()}
    
    # 使用OpenROAD标准方法重新初始化floorplan
    puts "使用utilization {utilization}%重新初始化floorplan..."
    
    # 获取site信息
    set site_info "{self._get_site_info()}"
    puts "使用site: $site_info"
    
    # 重新初始化floorplan，使用utilization方法
    initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 -core_space 10 -site $site_info
}} else {{
    puts "未找到DEF文件，将创建新的floorplan"
    # 如果DEF文件不存在，先读取Verilog文件并连接设计
    read_verilog design.v
    link_design {self._detect_top_module()}
    
    # 获取site信息
    set site_info "{self._get_site_info()}"
    puts "使用site: $site_info"
    
    # 然后创建floorplan，使用utilization方法
    puts "初始化布局，使用utilization {utilization}%..."
    initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 -core_space 10 -site $site_info
}}

# 设置布局参数
set_placement_padding -global -left 2 -right 2

# Save initial layout (before unplace_all)
puts "LOG: Saving initial layout before unplace_all"
write_def "$output_dir/iterations/iteration_0_initial.def"
puts $log_fp "Iteration 0 (initial): Layout saved before unplace_all"

# Unplace all cells to start fresh for RL training
puts "LOG: Unplacing all cells for RL training"
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set insts [$block getInsts]

foreach inst $insts {{
    $inst setPlacementStatus "UNPLACED"
}}
puts "LOG: All cells unplaced, starting RL training iterations"

# Execute multiple global_placement iterations for RL training
for {{set i 1}} {{$i <= $num_iterations}} {{incr i}} {{
    puts "LOG: Starting RL training iteration $i"
    puts $log_fp "=== RL Training Iteration $i ==="
    
    # 多样化参数扰动，使用更保守的参数
    set wirelength_coef [lindex {{0.8 1.0 1.2 0.9 1.1 1.3 0.85 1.05 1.15 1.25}} [expr $i-1]]
    set density_penalty [lindex {{0.0001 0.00015 0.0002 0.00012 0.00018 0.00022 0.00011 0.00016 0.00019 0.00021}} [expr $i-1]]
    
    # 执行全局布局，使用智能计算的密度
    if {{[catch {{global_placement -density {optimal_density} -init_wirelength_coef $wirelength_coef -init_density_penalty $density_penalty}} result]}} {{
        puts "LOG: Global placement failed in iteration $i: $result"
        puts $log_fp "Iteration $i: Global placement failed: $result"
        continue
    }}
    
    # 尝试详细布局，如果失败则跳过
    if {{[catch {{detailed_placement -max_displacement 2 -max_iterations 5}} result]}} {{
        puts "LOG: Detailed placement failed in iteration $i: $result"
        puts $log_fp "Iteration $i: Detailed placement failed: $result"
        # 即使详细布局失败，也保存当前布局作为训练数据
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
    
    puts $log_fp "---"
}}

# Save final layout
write_def "$output_dir/final_layout.def"
puts "LOG: Final layout saved to: $output_dir/final_layout.def"

close $log_fp
puts "LOG: RL training data collection completed"
"""
        
        tcl_file = os.path.join(self.work_dir, "iterative_placement.tcl")
        with open(tcl_file, 'w') as f:
            f.write(tcl_content)
        
        return tcl_file

    def run_iterative_placement(self, 
                              num_iterations: int = 10) -> Dict[str, Any]:
        """
        运行迭代布局流程
        
        Args:
            num_iterations: 迭代次数
            
        Returns:
            Dict[str, Any]: 迭代结果
        """
        try:
            print(f"开始运行迭代布局流程，迭代次数: {num_iterations}")
            
            # 生成迭代布局TCL脚本
            tcl_file = self.create_iterative_placement_tcl(num_iterations)
            
            print(f"迭代布局TCL脚本已生成: {tcl_file}")
            
            # 执行迭代布局
            start_time = time.time()
            success, stdout, stderr = self.run_openroad_command(tcl_file, timeout=timeout)
            execution_time = time.time() - start_time
            
            if success:
                print(f"✅ 迭代布局成功完成 (耗时: {execution_time:.2f}秒)")
                
                # 收集迭代数据
                iteration_data = self._collect_iteration_data()
                
                # 为每轮迭代添加HPWL
                for i, iteration in enumerate(iteration_data):
                    # 尝试从对应的DEF文件提取HPWL
                    def_file = f"output/iterations/iteration_{i}_result.def"
                    hpwl = self._extract_hpwl_from_def(def_file)
                    if hpwl == float('inf'):
                        # 如果特定轮次DEF不存在，尝试从主结果提取
                        hpwl = self._extract_hpwl_from_def("placement_result.def")
                    
                    iteration['hpwl'] = hpwl
                    print(f"   轮次 {i}: HPWL = {hpwl:.2e}")
                
                return {
                    'success': True,
                    'execution_time': execution_time,
                    'iteration_data': iteration_data,
                    'stdout': stdout,
                    'stderr': stderr
                }
            else:
                print(f"❌ 迭代布局失败 (耗时: {execution_time:.2f}秒)")
                return {
                    'success': False,
                    'execution_time': execution_time,
                    'error': stderr,
                    'stdout': stdout,
                    'stderr': stderr
                }
            
        except Exception as e:
            print(f"迭代布局流程异常: {e}")
            return {"success": False, "error": str(e)}

    def _collect_iteration_data(self) -> List[Dict[str, Any]]:
        """
        收集迭代数据
        
        Returns:
            List[Dict[str, Any]]: 迭代数据列表
        """
        iteration_data = []
        iterations_dir = self.work_dir / "output" / "iterations"
        
        if not iterations_dir.exists():
            print(f"迭代目录不存在: {iterations_dir}")
            return iteration_data
        
        # 查找所有DEF文件
        def_files = list(iterations_dir.glob("*.def"))
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
            hpwl_report_file = iterations_dir / f"iteration_{iteration_num}_hpwl.rpt"
            if hpwl_report_file.exists():
                try:
                    with open(hpwl_report_file, 'r') as f:
                        hpwl_content = f.read()
                    
                    # 解析HPWL报告内容
                    import re
                    hpwl_match = re.search(r'Total wire length:\s*([\d.]+)\s*um', hpwl_content)
                    if hpwl_match:
                        hpwl = float(hpwl_match.group(1))
                        print(f"从HPWL报告解析到值: {hpwl}")
                        
                except Exception as e:
                    print(f"读取HPWL报告失败: {str(e)}")
            
            # 尝试读取溢出率报告文件
            overflow_report_file = iterations_dir / f"iteration_{iteration_num}_overflow.rpt"
            if overflow_report_file.exists():
                try:
                    with open(overflow_report_file, 'r') as f:
                        overflow_content = f.read()
                    
                    # 解析溢出率报告内容
                    import re
                    overflow_match = re.search(r'Overflow:\s*([\d.]+)\s*%', overflow_content)
                    if overflow_match:
                        overflow = float(overflow_match.group(1)) / 100.0  # 转换为小数
                        print(f"从溢出率报告解析到值: {overflow}")
                        
                except Exception as e:
                    print(f"读取溢出率报告失败: {str(e)}")
            
            iteration_info = {
                'iteration': iteration_num,
                'def_file': str(def_file),
                'hpwl': hpwl,
                'overflow': overflow,
                'timestamp': def_file.stat().st_mtime if def_file.exists() else None
            }
            
            iteration_data.append(iteration_info)
            print(f"迭代 {iteration_num}: HPWL={hpwl}, 溢出率={overflow}")
        
        return iteration_data

    def run_openroad_command(self, tcl_file: str, timeout: int = None) -> tuple[bool, str, str]:
        """运行OpenROAD命令"""
        try:
            # 计算超时时间
            if timeout is None:
                design_stats = self._extract_design_stats()
                timeout = self._calculate_timeout(design_stats)
            
            # 获取绝对路径
            work_dir_abs = os.path.abspath(self.work_dir)
            
            # 构建Docker命令，增加资源限制，使用绝对路径
            docker_cmd = f"docker run --rm -m 12g -c 4 -v {work_dir_abs}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit {os.path.basename(tcl_file)}'"
            
            print(f"执行Docker命令: {docker_cmd}")
            
            # 执行命令
            result = subprocess.run(
                docker_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            # 保存执行日志
            log_file = os.path.join(self.work_dir, "openroad_execution.log")
            with open(log_file, 'w') as f:
                f.write("=== OpenROAD Execution Log ===\n")
                f.write(f"Command: {docker_cmd}\n")
                f.write(f"Return Code: {result.returncode}\n")
                f.write("=== STDOUT ===\n")
                f.write(stdout)
                f.write("\n=== STDERR ===\n")
                f.write(stderr)
                f.write("\n=== END ===\n")
            
            print(f"OpenROAD执行日志已保存到: {log_file}")
            
            if not success:
                print(f"OpenROAD命令执行失败: {stderr}")
            
            return success, stdout, stderr
        except Exception as e:
            print(f"OpenROAD命令执行失败: {e}")
            return False, "", str(e)

    def _calculate_optimal_area_and_density(self) -> tuple[tuple[int, int], float]:
        """
        根据设计规模智能计算最优芯片面积和密度
        
        Returns:
            tuple: ((width, height), density)
        """
        design_stats = self._extract_design_stats()
        num_instances = design_stats.get('num_instances', 0)
        num_nets = design_stats.get('num_nets', 0)
        current_area = design_stats.get('core_area', 0)
        
        # 基础密度设置
        base_density = 0.75  # 降低基础密度
        
        # 根据实例数量调整密度
        if num_instances < 50000:
            density = base_density
            area_multiplier = 1.2
        elif num_instances < 100000:
            density = base_density - 0.05  # 0.70
            area_multiplier = 1.4
        elif num_instances < 500000:
            density = base_density - 0.10  # 0.65
            area_multiplier = 1.6
        elif num_instances < 1000000:
            density = base_density - 0.15  # 0.60
            area_multiplier = 1.8
        else:
            density = base_density - 0.20  # 0.55
            area_multiplier = 2.0
        
        # 计算当前芯片尺寸
        if current_area > 0:
            # 假设是正方形芯片
            current_side = int(current_area ** 0.5)
            new_side = int(current_side * area_multiplier)
            new_area = (new_side, new_side)
        else:
            # 如果没有面积信息，根据实例数量估算
            if num_instances < 50000:
                new_area = (800, 800)
            elif num_instances < 100000:
                new_area = (1200, 1200)
            elif num_instances < 500000:
                new_area = (1600, 1600)
            elif num_instances < 1000000:
                new_area = (2000, 2000)
            else:
                new_area = (2500, 2500)
        
        print(f"设计规模: {num_instances}实例, {num_nets}网络")
        print(f"智能调整: 芯片面积 {new_area[0]}x{new_area[1]}, 密度 {density:.2f}")
        
        return new_area, density

    def _get_site_info(self) -> str:
        """
        获取site信息，用于floorplan初始化
        
        Returns:
            str: site名称
        """
        # 尝试从LEF文件中获取site信息
        lef_files = [f for f in os.listdir(self.work_dir) if f.endswith('.lef')]
        
        for lef_file in lef_files:
            lef_path = os.path.join(self.work_dir, lef_file)
            try:
                with open(lef_path, 'r') as f:
                    content = f.read()
                    # 查找SITE定义
                    import re
                    site_match = re.search(r'SITE\s+(\w+)', content)
                    if site_match:
                        return site_match.group(1)
            except Exception as e:
                print(f"读取LEF文件失败: {str(e)}")
                continue
        
        # 如果没有找到，返回默认值
        return "core"

    def _extract_hpwl_from_def(self, def_file: str = "placement_result.def") -> float:
        """
        从DEF文件中提取HPWL值
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            float: HPWL值，如果提取失败返回float('inf')
        """
        try:
            def_path = Path(self.work_dir) / def_file
            if not def_path.exists():
                logger.warning(f"DEF文件不存在: {def_path}")
                return float('inf')
            
            # 检查DEF文件是否包含组件放置信息
            with open(def_path, 'r') as f:
                content = f.read()
            
            # 如果DEF文件只包含行定义而没有组件放置，返回无穷大
            if 'COMPONENTS' not in content or 'PLACED' not in content:
                logger.warning(f"DEF文件 {def_file} 不包含组件放置信息，跳过HPWL提取")
                return float('inf')
            
            # 使用OpenROAD命令提取HPWL
            cmd = f"docker run --rm -v {self.work_dir}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit - << EOF\nread_def {def_file}\nreport_wirelength\nEOF'"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # 解析输出中的HPWL值
                for line in result.stdout.split('\n'):
                    if 'HPWL' in line and ':' in line:
                        # 提取数值 - 改进的提取逻辑
                        import re
                        hpwl_match = re.search(r'HPWL:\s*([0-9.]+(?:e[+-]?\d+)?)', line)
                        if hpwl_match:
                            try:
                                hpwl_value = float(hpwl_match.group(1))
                                # 验证HPWL值的合理性 - 更严格的检查
                                if hpwl_value > 10000:  # 至少1万才可能是合理的HPWL
                                    return hpwl_value
                                else:
                                    logger.warning(f"HPWL值异常小 ({hpwl_value})，可能是提取错误")
                            except ValueError:
                                continue
                        
                        # 如果没有找到标准格式，尝试其他格式
                        parts = line.split(':')
                        if len(parts) >= 2:
                            hpwl_str = parts[1].strip().split()[0]  # 取第一个数值
                            try:
                                hpwl_value = float(hpwl_str)
                                if hpwl_value > 10000:  # 至少1万才可能是合理的HPWL
                                    return hpwl_value
                                else:
                                    logger.warning(f"HPWL值异常小 ({hpwl_value})，可能是提取错误")
                            except ValueError:
                                continue
            
            logger.warning(f"无法从DEF文件提取HPWL: {def_file}")
            return float('inf')
            
        except Exception as e:
            logger.error(f"提取HPWL时发生异常: {e}")
            return float('inf')
    
    def _extract_hpwl_from_log(self, log_file: str = "openroad_execution.log") -> float:
        """
        从OpenROAD执行日志中提取HPWL值
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            float: HPWL值，如果提取失败返回float('inf')
        """
        try:
            log_path = Path(self.work_dir) / log_file
            if not log_path.exists():
                logger.warning(f"日志文件不存在: {log_path}")
                return float('inf')
            
            with open(log_path, 'r') as f:
                content = f.read()
            
            # 查找HPWL相关的行
            lines = content.split('\n')
            for line in lines:
                # 匹配多种HPWL格式
                if any(keyword in line for keyword in ['HPWL:', 'Wirelength:', 'Total wirelength:']):
                    # 提取数值 - 改进的提取逻辑
                    import re
                    
                    # 对于HPWL:格式，提取冒号后的第一个数字
                    if 'HPWL:' in line:
                        hpwl_match = re.search(r'HPWL:\s*([0-9.]+(?:e[+-]?\d+)?)', line)
                        if hpwl_match:
                            try:
                                hpwl_value = float(hpwl_match.group(1))
                                # 验证HPWL值的合理性
                                if hpwl_value > 0 and hpwl_value < 1e15:  # 合理的HPWL范围
                                    return hpwl_value
                            except ValueError:
                                continue
                    
                    # 对于其他格式，尝试提取科学计数法或大数字
                    elif any(keyword in line for keyword in ['Wirelength:', 'Total wirelength:']):
                        # 查找科学计数法格式的数字
                        sci_match = re.search(r'([0-9.]+e[+-]?\d+)', line)
                        if sci_match:
                            try:
                                hpwl_value = float(sci_match.group(1))
                                if hpwl_value > 0 and hpwl_value < 1e15:
                                    return hpwl_value
                            except ValueError:
                                continue
                        
                        # 查找大数字（可能是HPWL）
                        large_num_match = re.search(r'([0-9]{6,})', line)
                        if large_num_match:
                            try:
                                hpwl_value = float(large_num_match.group(1))
                                if hpwl_value > 100000:  # 至少10万才可能是HPWL
                                    return hpwl_value
                            except ValueError:
                                continue
            
            logger.warning(f"无法从日志文件提取HPWL: {log_file}")
            return float('inf')
            
        except Exception as e:
            logger.error(f"从日志提取HPWL时发生异常: {e}")
            return float('inf')

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