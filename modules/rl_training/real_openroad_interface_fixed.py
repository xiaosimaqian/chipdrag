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
        tcl_script = f"""
# OpenROAD布局优化脚本 (智能参数版本)
# 设计: {self.verilog_file.name}
# 参数: density_target={density_target}, die_size={die_size}x{die_size}, core_size={core_size}x{core_size}

# 完全重置数据库
if {{[info exists ::ord::db]}} {{
    ord::reset_db
}}

# 读取LEF文件 - 先读取技术LEF，再读取单元LEF
read_lef {self.tech_lef.name}
read_lef {self.cells_lef.name}

# 读取Liberty文件（如果存在）
{lib_read_cmds_str}

# 读取Verilog文件
read_verilog {self.verilog_file.name}

# 连接设计
link_design {top_module}

# 设置工艺参数
set tech [ord::get_db_tech]

# 使用动态计算的布局参数
puts "初始化布局 (die: {die_size}x{die_size}, core: {core_size}x{core_size})..."
initialize_floorplan -die_area "0 0 {die_size} {die_size}" -core_area "10 10 {core_size} {core_size}" -site core

# 设置布局参数 - 使用更保守的设置
set_placement_padding -global -left 1 -right 1

# 全局布局 - 使用动态密度目标
puts "开始全局布局 (密度目标: {density_target})..."
global_placement -density {density_target}

# 详细布局
puts "开始详细布局..."
detailed_placement

# 输出结果
write_def placement_result.def
write_verilog placement_result.v

puts "布局优化完成"
puts "输出文件: placement_result.def, placement_result.v"
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
            
            # 执行OpenROAD命令
            start_time = time.time()
            result = subprocess.run([
                'docker', 'run', '--rm',
                '-v', f'{self.work_dir}:/workspace',
                '-w', '/workspace',
                'openroad/flow-ubuntu22.04-builder:21e414',
                'bash', '-c',
                f'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -no_init -no_splash -exit openroad_script.tcl'
            ], capture_output=True, text=True, timeout=timeout)
            
            execution_time = time.time() - start_time
            
            # 检查结果
            success = result.returncode == 0
            
            # 提取结果信息
            wirelength = None
            area = None
            
            if success:
                # 尝试从输出中提取线长和面积信息
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'HPWL' in line:
                        try:
                            wirelength = float(line.split()[-1])
                        except:
                            pass
                    elif 'Core area' in line:
                        try:
                            area = float(line.split()[-2])
                        except:
                            pass
            
            return {
                'success': success,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'wirelength': wirelength,
                'area': area,
                'design_stats': design_stats,
                'optimal_params': optimal_params
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'execution_time': timeout,
                'return_code': -1,
                'stdout': '',
                'stderr': '执行超时',
                'wirelength': None,
                'area': None,
                'design_stats': design_stats if 'design_stats' in locals() else {},
                'optimal_params': optimal_params if 'optimal_params' in locals() else {}
            }
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'wirelength': None,
                'area': None,
                'design_stats': design_stats if 'design_stats' in locals() else {},
                'optimal_params': optimal_params if 'optimal_params' in locals() else {}
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
                        logger.info(f"从LEF文件中提取到站点: {site_name}")
                        return site_name
            
            # 如果无法从LEF文件中提取，使用ISPD基准测试的默认站点
            logger.info("使用ISPD基准测试默认站点: core")
            return "core"
            
        except Exception as e:
            logger.warning(f"站点提取失败: {e}，使用默认站点: core")
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
        根据设计规模计算智能超时时间
        
        Args:
            design_stats: 设计统计信息
            
        Returns:
            超时时间（秒）
        """
        # 基础超时时间
        base_timeout = 300  # 5分钟
        
        # 从设计统计中提取关键指标
        num_instances = design_stats.get('num_instances', 0)
        num_nets = design_stats.get('num_nets', 0)
        num_pins = design_stats.get('num_pins', 0)
        core_area = design_stats.get('core_area', 0)
        
        # 根据实例数量调整超时时间
        if num_instances > 0:
            # 每1000个实例增加1分钟
            instance_factor = max(1.0, num_instances / 1000.0)
            timeout = int(base_timeout * instance_factor)
        else:
            timeout = base_timeout
        
        # 根据网络数量进一步调整
        if num_nets > 0:
            # 每1000个网络增加30秒
            net_factor = max(1.0, num_nets / 1000.0 * 0.5)
            timeout = int(timeout * net_factor)
        
        # 根据核心面积调整
        if core_area > 0:
            # 每100,000 um²增加1分钟
            area_factor = max(1.0, core_area / 100000.0)
            timeout = int(timeout * area_factor)
        
        # 设置最小和最大超时限制
        min_timeout = 120   # 2分钟
        max_timeout = 1800  # 30分钟
        
        timeout = max(min_timeout, min(timeout, max_timeout))
        
        logger.info(f"设计规模: {num_instances}实例, {num_nets}网络, {core_area:.0f}um²面积")
        logger.info(f"计算超时时间: {timeout}秒")
        
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
                        stats['core_area'] = (x2 - x1) * (y2 - y1)
            
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
            logger.warning(f"提取设计统计信息时出错: {e}")
        
        return stats

    def _calculate_optimal_parameters(self, design_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据设计规模计算最优参数
        
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
        
        # 根据实例数量调整密度目标
        if num_instances > 100000:  # 超大型设计
            density_target = 0.85  # 提高密度目标
            die_size = 1200
            core_size = 1190
        elif num_instances > 50000:  # 大型设计
            density_target = 0.80
            die_size = 1000
            core_size = 990
        elif num_instances > 20000:  # 中型设计
            density_target = 0.75
            die_size = 900
            core_size = 890
        else:  # 小型设计
            density_target = base_density
            die_size = base_die_size
            core_size = base_core_size
        
        # 根据网络数量进一步调整
        if num_nets > 100000:
            density_target = min(0.90, density_target + 0.05)
            die_size = max(die_size, 1400)
            core_size = die_size - 10
        
        # 根据核心面积调整
        if core_area > 500000:  # 500,000 um²
            density_target = min(0.90, density_target + 0.05)
            die_size = max(die_size, 1200)
            core_size = die_size - 10
        
        return {
            'density_target': density_target,
            'die_size': die_size,
            'core_size': core_size,
            'wirelength_weight': 1.0,
            'density_weight': 1.0
        }

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