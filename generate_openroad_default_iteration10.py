#!/usr/bin/env python3
"""
生成OpenROAD默认参数下的第10轮DEF文件
用于补充缺失的iteration_10.def文件
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DefaultIterationGenerator:
    def __init__(self, benchmark_dir: str = "data/designs/ispd_2015_contest_benchmark"):
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path("results/iterations")
        self.results_dir.mkdir(exist_ok=True)
        
    def get_design_directories(self):
        """获取所有设计目录"""
        design_dirs = []
        for item in self.benchmark_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                design_dirs.append(item)
        return sorted(design_dirs)
    
    def create_default_iteration_tcl(self, design_dir: Path, output_dir: Path) -> str:
        """创建默认参数的迭代TCL脚本"""
        design_name = design_dir.name
        
        # 查找设计文件
        lef_files = list(design_dir.glob("*.lef"))
        verilog_files = list(design_dir.glob("*.v"))
        
        if not lef_files or not verilog_files:
            raise ValueError(f"设计 {design_name} 缺少LEF或Verilog文件")
        
        lef_file = lef_files[0].name
        verilog_file = verilog_files[0].name
        
        # 检测顶层模块
        top_module = self._detect_top_module(design_dir / verilog_file)
        
        # 计算面积和密度
        die_size, core_size, utilization = self._calculate_area_and_density(design_dir)
        
        # 获取site信息
        site_info = self._get_site_info(design_dir)
        
        tcl_content = f"""#!/usr/bin/env tclsh

# OpenROAD默认参数迭代脚本
# 设计: {design_name}
# 目标: 生成iteration_10.def（OpenROAD默认参数）

set output_dir "{output_dir}"
file mkdir "$output_dir/iterations"

# 设置日志文件
set log_fp [open "$output_dir/default_iteration.log" w]
puts $log_fp "=== OpenROAD Default Parameters Iteration ==="
puts $log_fp "Design: {design_name}"
puts $log_fp "Target: iteration_10.def"

# 读取设计文件
read_lef "{lef_file}"
read_def "{design_name}.def"
read_verilog "{verilog_file}"

# 链接设计
link_design {top_module}

# 设置技术信息
set tech [ord::get_db_tech]
set site_info "{site_info}"
puts "使用site: $site_info"

# 初始化布局
puts "初始化布局，使用utilization {utilization}%..."
initialize_floorplan -utilization {utilization} -aspect_ratio 1.0 -core_space 10 -site $site_info

set_placement_padding -global -left 2 -right 2

# 保存初始布局
write_def "$output_dir/iterations/iteration_0_initial.def"
puts $log_fp "Iteration 0: Initial layout saved"

# 运行10次迭代，使用OpenROAD默认参数
for {{set iteration 1}} {{$iteration <= 10}} {{incr iteration}} {{
    puts "LOG: Starting iteration $iteration with default parameters"
    puts $log_fp "=== Iteration $iteration ==="
    
    # 全局布局（使用默认参数）
    if {{[catch {{global_placement}} result]}} {{
        puts "LOG: Global placement failed in iteration $iteration: $result"
        puts $log_fp "Iteration $iteration: Global placement failed: $result"
    }} else {{
        puts "LOG: Global placement completed for iteration $iteration"
        puts $log_fp "Iteration $iteration: Global placement completed"
    }}
    
    # 详细布局
    if {{[catch {{detailed_placement}} result]}} {{
        puts "LOG: Detailed placement failed in iteration $iteration: $result"
        puts $log_fp "Iteration $iteration: Detailed placement failed: $result"
    }} else {{
        puts "LOG: Detailed placement completed for iteration $iteration"
        puts $log_fp "Iteration $iteration: Detailed placement completed"
    }}
    
    # 引脚布局
    if {{[catch {{place_pins -hor_layers 2 -ver_layers 2}} result]}} {{
        puts "LOG: Pin placement failed in iteration $iteration: $result"
        puts $log_fp "Iteration $iteration: Pin placement failed: $result"
    }} else {{
        puts "LOG: Pin placement completed for iteration $iteration"
        puts $log_fp "Iteration $iteration: Pin placement completed"
    }}
    
    # 保存当前迭代的布局
    set def_filename "$output_dir/iterations/iteration_${{iteration}}.def"
    write_def $def_filename
    puts "LOG: Layout saved to: $def_filename"
    puts $log_fp "DEF file: $def_filename"
    
    # 收集指标
    set hpwl_report_file "$output_dir/iterations/iteration_${{iteration}}_hpwl.rpt"
    if {{[catch {{report_wire_length}} result]}} {{
        puts "LOG: Cannot get HPWL information: $result"
        puts $log_fp "Iteration $iteration: HPWL=unavailable"
    }} else {{
        set hpwl_fp [open $hpwl_report_file w]
        puts $hpwl_fp $result
        close $hpwl_fp
        puts "LOG: HPWL report saved to: $hpwl_report_file"
        puts $log_fp "Iteration $iteration: HPWL report saved"
    }}
    
    set overflow_report_file "$output_dir/iterations/iteration_${{iteration}}_overflow.rpt"
    if {{[catch {{report_placement}} result]}} {{
        puts "LOG: Cannot get placement information: $result"
        puts $log_fp "Iteration $iteration: Placement info=unavailable"
    }} else {{
        set overflow_fp [open $overflow_report_file w]
        puts $overflow_fp $result
        close $overflow_fp
        puts "LOG: Placement report saved to: $overflow_report_file"
        puts $log_fp "Iteration $iteration: Placement report saved"
    }}
    
    puts $log_fp "---"
}}

# 保存最终布局
write_def "$output_dir/final_layout_default.def"
puts "LOG: Final layout saved to: $output_dir/final_layout_default.def"
close $log_fp
puts "LOG: Default iteration generation completed"
"""
        
        tcl_file = design_dir / "default_iteration.tcl"
        with open(tcl_file, 'w') as f:
            f.write(tcl_content)
        
        return str(tcl_file)
    
    def _detect_top_module(self, verilog_file: Path) -> str:
        """检测顶层模块名"""
        try:
            with open(verilog_file, 'r') as f:
                content = f.read()
            
            # 简单的模块名检测
            import re
            module_match = re.search(r'module\s+(\w+)', content)
            if module_match:
                return module_match.group(1)
            else:
                return "top"
        except Exception as e:
            logger.warning(f"无法检测顶层模块，使用默认值: {e}")
            return "top"
    
    def _calculate_area_and_density(self, design_dir: Path) -> tuple[int, int, float]:
        """计算面积和密度"""
        # 简化的计算，可以根据需要调整
        return 800, 790, 70.0
    
    def _get_site_info(self, design_dir: Path) -> str:
        """获取site信息"""
        # 检查是否有tech.lef文件
        tech_lef = design_dir / "tech.lef"
        if tech_lef.exists():
            try:
                with open(tech_lef, 'r') as f:
                    content = f.read()
                
                import re
                site_match = re.search(r'SITE\s+(\w+)', content)
                if site_match:
                    return site_match.group(1)
            except Exception as e:
                logger.warning(f"无法从tech.lef解析site信息: {e}")
        
        # 默认使用core
        return "core"
    
    def run_openroad_command(self, tcl_file: str, timeout: int = 3600) -> tuple[bool, str, str]:
        """运行OpenROAD命令"""
        try:
            # 切换到设计目录
            design_dir = Path(tcl_file).parent
            os.chdir(design_dir)
            
            # 运行Docker命令
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{design_dir.absolute()}:/workspace",
                "-w", "/workspace",
                "--memory", "8g",
                "--cpus", "4",
                "openroad/openroad:latest",
                "openroad", "-exit", tcl_file
            ]
            
            logger.info(f"运行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"命令超时: {timeout}秒")
            return False, "", "Command timeout"
        except Exception as e:
            logger.error(f"运行命令失败: {e}")
            return False, "", str(e)
    
    def generate_default_iteration(self, design_dir: Path) -> dict:
        """为单个设计生成默认参数的迭代文件"""
        design_name = design_dir.name
        logger.info(f"开始为设计 {design_name} 生成默认参数迭代文件")
        
        try:
            # 创建输出目录
            output_dir = design_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            # 创建TCL脚本
            tcl_file = self.create_default_iteration_tcl(design_dir, output_dir)
            logger.info(f"TCL脚本已生成: {tcl_file}")
            
            # 运行OpenROAD
            start_time = time.time()
            success, stdout, stderr = self.run_openroad_command(tcl_file)
            execution_time = time.time() - start_time
            
            if success:
                # 检查是否生成了目标文件
                target_file = output_dir / "iterations" / "iteration_10.def"
                if target_file.exists():
                    logger.info(f"设计 {design_name} 成功生成默认参数迭代文件")
                    return {
                        "design_name": design_name,
                        "success": True,
                        "execution_time": execution_time,
                        "target_file": str(target_file),
                        "stdout": stdout,
                        "stderr": stderr
                    }
                else:
                    logger.warning(f"设计 {design_name} 未生成目标文件")
                    return {
                        "design_name": design_name,
                        "success": False,
                        "error": "目标文件未生成",
                        "execution_time": execution_time,
                        "stdout": stdout,
                        "stderr": stderr
                    }
            else:
                logger.error(f"设计 {design_name} 运行失败: {stderr}")
                return {
                    "design_name": design_name,
                    "success": False,
                    "error": stderr,
                    "execution_time": execution_time,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
        except Exception as e:
            logger.error(f"设计 {design_name} 处理异常: {e}")
            return {
                "design_name": design_name,
                "success": False,
                "error": str(e)
            }
    
    def run_batch_generation(self, max_workers: int = 4):
        """批量生成默认参数迭代文件"""
        design_dirs = self.get_design_directories()
        logger.info(f"找到 {len(design_dirs)} 个设计目录")
        
        results = []
        
        # 串行处理，避免资源冲突
        for design_dir in design_dirs:
            result = self.generate_default_iteration(design_dir)
            results.append(result)
            
            # 保存单个结果
            result_file = self.results_dir / f"{design_dir.name}_default_iteration_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"设计 {design_dir.name} 处理完成")
        
        # 生成报告
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results: list):
        """生成报告"""
        report_file = self.results_dir / "default_iteration_report.md"
        
        total_designs = len(results)
        successful_designs = sum(1 for r in results if r.get("success", False))
        failed_designs = total_designs - successful_designs
        
        report_content = f"""# OpenROAD默认参数迭代文件生成报告

## 生成概览
- **总设计数量**: {total_designs}
- **成功设计数量**: {successful_designs}
- **失败设计数量**: {failed_designs}
- **成功率**: {successful_designs/total_designs*100:.1f}%

## 详细结果
"""
        
        for result in results:
            design_name = result["design_name"]
            success = result.get("success", False)
            error = result.get("error", "")
            execution_time = result.get("execution_time", 0)
            
            report_content += f"""
### {design_name}
- **状态**: {'成功' if success else '失败'}
- **耗时**: {execution_time:.2f}秒
"""
            
            if not success and error:
                report_content += f"- **错误**: {error}\n"
            
            if success:
                target_file = result.get("target_file", "")
                report_content += f"- **生成文件**: {target_file}\n"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"报告已生成: {report_file}")

def main():
    generator = DefaultIterationGenerator()
    results = generator.run_batch_generation()
    
    successful = sum(1 for r in results if r.get("success", False))
    logger.info(f"生成完成: {successful}/{len(results)} 个设计成功")

if __name__ == "__main__":
    main() 