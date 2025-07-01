#!/usr/bin/env python3
"""
论文HPWL对比实验脚本
收集三组真实HPWL数据：
1. 极差布局HPWL (iteration_0_initial.def)
2. OpenROAD默认布局HPWL (iteration_10.def) 
3. ChipDRAG优化布局HPWL (iteration_10_rl_training.def)
"""

import os
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperHPWLComparisonExperiment:
    """论文HPWL对比实验系统"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data/designs/ispd_2015_contest_benchmark"
        self.results_dir = self.base_dir / "paper_hpwl_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 实验配置
        self.experiment_config = {
            'designs': [
                'mgc_des_perf_a', 'mgc_des_perf_1', 'mgc_des_perf_b',
                'mgc_edit_dist_a', 'mgc_fft_1', 'mgc_fft_2', 
                'mgc_fft_a', 'mgc_fft_b', 'mgc_matrix_mult_1',
                'mgc_matrix_mult_a', 'mgc_matrix_mult_b',
                'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b'
            ],
            'hpwl_script': self.base_dir / "calculate_hpwl.py"
        }
        
        logger.info(f"论文HPWL对比实验系统初始化完成")
        logger.info(f"目标设计: {len(self.experiment_config['designs'])}个")
    
    def extract_hpwl_from_def(self, def_file: Path) -> Optional[float]:
        """从DEF文件中提取HPWL值"""
        try:
            if not def_file.exists():
                logger.warning(f"DEF文件不存在: {def_file}")
                return None
            
            # 使用HPWL计算脚本
            result = subprocess.run([
                'python', str(self.experiment_config['hpwl_script']), str(def_file)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # 解析输出
                for line in result.stdout.split('\n'):
                    if line.startswith('Total HPWL:'):
                        hpwl_str = line.split(':')[1].strip()
                        hpwl_value = float(hpwl_str)
                        # 转换为微米单位
                        hpwl_microns = hpwl_value / 1000.0
                        return hpwl_microns
            
            logger.error(f"HPWL提取失败: {result.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"提取HPWL时出错: {e}")
            return None
    
    def collect_three_group_hpwl(self) -> Dict[str, Dict[str, Any]]:
        """收集三组HPWL数据"""
        logger.info("开始收集三组HPWL数据...")
        
        results = {}
        
        for design_name in self.experiment_config['designs']:
            design_dir = self.data_dir / design_name
            if not design_dir.exists():
                logger.warning(f"设计目录不存在: {design_dir}")
                continue
            
            logger.info(f"处理设计: {design_name}")
            
            # 查找三种DEF文件
            iterations_dir = design_dir / "output" / "iterations"
            if not iterations_dir.exists():
                logger.warning(f"迭代目录不存在: {iterations_dir}")
                continue
            
            # 1. 极差布局HPWL (iteration_0_initial.def)
            worst_def = iterations_dir / "iteration_0_initial.def"
            worst_hpwl = self.extract_hpwl_from_def(worst_def)
            
            # 2. OpenROAD默认布局HPWL (iteration_10.def)
            default_def = iterations_dir / "iteration_10.def"
            default_hpwl = self.extract_hpwl_from_def(default_def)
            
            # 3. ChipDRAG优化布局HPWL (iteration_10_rl_training.def)
            optimized_def = iterations_dir / "iteration_10_rl_training.def"
            optimized_hpwl = self.extract_hpwl_from_def(optimized_def)
            
            # 记录结果
            results[design_name] = {
                'worst_hpwl': worst_hpwl,
                'default_hpwl': default_hpwl,
                'optimized_hpwl': optimized_hpwl,
                'worst_def_exists': worst_def.exists(),
                'default_def_exists': default_def.exists(),
                'optimized_def_exists': optimized_def.exists()
            }
            
            # 计算提升率
            if worst_hpwl and default_hpwl and optimized_hpwl:
                default_improvement = ((worst_hpwl - default_hpwl) / worst_hpwl) * 100
                optimized_improvement = ((worst_hpwl - optimized_hpwl) / worst_hpwl) * 100
                chipdrag_vs_default = ((default_hpwl - optimized_hpwl) / default_hpwl) * 100
                
                results[design_name].update({
                    'default_improvement_pct': default_improvement,
                    'optimized_improvement_pct': optimized_improvement,
                    'chipdrag_vs_default_pct': chipdrag_vs_default
                })
                
                logger.info(f"  {design_name}: 极差={worst_hpwl:.2e}, 默认={default_hpwl:.2e}, 优化={optimized_hpwl:.2e}")
                logger.info(f"    默认提升: {default_improvement:.2f}%, ChipDRAG提升: {optimized_improvement:.2f}%, ChipDRAG vs 默认: {chipdrag_vs_default:.2f}%")
            else:
                logger.warning(f"  {design_name}: 部分HPWL数据缺失")
        
        return results
    
    def generate_missing_default_defs(self) -> Dict[str, bool]:
        """为缺失的OpenROAD默认DEF文件生成TCL脚本"""
        logger.info("检查并生成缺失的OpenROAD默认DEF文件...")
        
        missing_results = {}
        
        for design_name in self.experiment_config['designs']:
            design_dir = self.data_dir / design_name
            iterations_dir = design_dir / "output" / "iterations"
            default_def = iterations_dir / "iteration_10.def"
            
            if not default_def.exists():
                logger.info(f"为 {design_name} 生成OpenROAD默认DEF文件...")
                success = self._generate_default_def(design_dir, design_name)
                missing_results[design_name] = success
            else:
                missing_results[design_name] = True
        
        return missing_results
    
    def _generate_default_def(self, design_dir: Path, design_name: str) -> bool:
        """为单个设计生成OpenROAD默认DEF文件"""
        try:
            # 创建默认参数的TCL脚本
            tcl_content = self._create_default_placement_tcl(design_dir, design_name)
            tcl_file = design_dir / "default_placement.tcl"
            
            with open(tcl_file, 'w') as f:
                f.write(tcl_content)
            
            # 运行OpenROAD命令
            work_dir_abs = design_dir.absolute()
            docker_cmd = f"docker run --rm -m 16g -c 8 -e OPENROAD_NUM_THREADS=8 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -v {work_dir_abs}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit default_placement.tcl'"
            
            logger.info(f"执行默认布局: {design_name}")
            result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=7200)
            
            if result.returncode == 0:
                # 检查是否生成了iteration_10.def
                output_def = design_dir / "output" / "iterations" / "iteration_10.def"
                if output_def.exists():
                    logger.info(f"✅ {design_name} 默认DEF文件生成成功")
                    return True
                else:
                    logger.error(f"❌ {design_name} 默认DEF文件未生成")
                    return False
            else:
                logger.error(f"❌ {design_name} 默认布局失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {design_name} 生成默认DEF时出错: {e}")
            return False
    
    def _create_default_placement_tcl(self, design_dir: Path, design_name: str) -> str:
        """创建OpenROAD默认布局TCL脚本"""
        # 自动检测文件
        lef_files = list(design_dir.glob("*.lef"))
        verilog_files = list(design_dir.glob("*.v"))
        def_files = list(design_dir.glob("*.def"))
        
        # 选择文件
        lef_file = lef_files[0].name if lef_files else "tech.lef"
        verilog_file = verilog_files[0].name if verilog_files else "design.v"
        
        # 优先选择floorplan.def作为初始布局
        def_file = None
        for def_path in def_files:
            if "floorplan" in def_path.name.lower():
                def_file = def_path.name
                break
        if not def_file and def_files:
            def_file = def_files[0].name
        
        # 生成LEF读取命令
        lef_read_cmds = []
        for lef in lef_files:
            lef_read_cmds.append(f'read_lef "{lef.name}"')
        lef_read_cmds_str = "\n".join(lef_read_cmds)
        
        tcl_script = f"""# OpenROAD默认布局脚本 - {design_name}
set output_dir "/workspace/output"
file mkdir $output_dir
file mkdir "$output_dir/iterations"

# 读取设计文件
{lef_read_cmds_str}
read_verilog {verilog_file}
read_def {def_file}

# 保存初始布局
write_def "$output_dir/iterations/iteration_0_initial.def"

# 卸载所有单元
unplace_all

# 执行10轮默认参数的全局布局
for {{set i 1}} {{$i <= 10}} {{incr i}} {{
    puts "执行第$i轮默认全局布局"
    global_placement
    detailed_placement
    
    # 保存当前布局
    write_def "$output_dir/iterations/iteration_${{i}}.def"
    
    # 报告HPWL
    if {{[catch {{report_wire_length}} result]}} {{
        puts "无法获取HPWL信息"
    }} else {{
        puts "第$i轮HPWL: $result"
    }}
}}

puts "默认布局完成"
"""
        return tcl_script
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成对比报告"""
        logger.info("生成HPWL对比报告...")
        
        # 统计信息
        total_designs = len(results)
        complete_designs = sum(1 for r in results.values() 
                             if r.get('worst_hpwl') and r.get('default_hpwl') and r.get('optimized_hpwl'))
        
        # 计算平均提升率
        improvements = []
        for design_name, data in results.items():
            if data.get('default_improvement_pct') and data.get('optimized_improvement_pct'):
                improvements.append({
                    'design': design_name,
                    'default_improvement': data['default_improvement_pct'],
                    'optimized_improvement': data['optimized_improvement_pct'],
                    'chipdrag_vs_default': data['chipdrag_vs_default_pct']
                })
        
        if improvements:
            avg_default_improvement = sum(i['default_improvement'] for i in improvements) / len(improvements)
            avg_optimized_improvement = sum(i['optimized_improvement'] for i in improvements) / len(improvements)
            avg_chipdrag_vs_default = sum(i['chipdrag_vs_default'] for i in improvements) / len(improvements)
        else:
            avg_default_improvement = avg_optimized_improvement = avg_chipdrag_vs_default = 0
        
        report = {
            'summary': {
                'total_designs': total_designs,
                'complete_designs': complete_designs,
                'completion_rate': complete_designs / total_designs if total_designs > 0 else 0,
                'avg_default_improvement': avg_default_improvement,
                'avg_optimized_improvement': avg_optimized_improvement,
                'avg_chipdrag_vs_default': avg_chipdrag_vs_default
            },
            'detailed_results': results,
            'improvements': improvements
        }
        
        return report
    
    def save_results(self, results: Dict[str, Any], report: Dict[str, Any]):
        """保存实验结果"""
        # 保存详细结果
        results_file = self.results_dir / "hpwl_comparison_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存报告
        report_file = self.results_dir / "hpwl_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成CSV文件
        csv_data = []
        for design_name, data in results.items():
            csv_data.append({
                'Design': design_name,
                'Worst_HPWL': data.get('worst_hpwl'),
                'Default_HPWL': data.get('default_hpwl'),
                'Optimized_HPWL': data.get('optimized_hpwl'),
                'Default_Improvement_%': data.get('default_improvement_pct'),
                'Optimized_Improvement_%': data.get('optimized_improvement_pct'),
                'ChipDRAG_vs_Default_%': data.get('chipdrag_vs_default_pct')
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / "hpwl_comparison_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"结果已保存到: {self.results_dir}")
    
    def generate_visualizations(self, report: Dict[str, Any]):
        """生成可视化图表"""
        logger.info("生成可视化图表...")
        
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        improvements = report.get('improvements', [])
        if not improvements:
            logger.warning("没有完整的改进数据，跳过可视化")
            return
        
        # 1. HPWL对比柱状图
        self._plot_hpwl_comparison(improvements, viz_dir)
        
        # 2. 提升率对比图
        self._plot_improvement_comparison(improvements, viz_dir)
        
        # 3. ChipDRAG vs 默认提升率分布
        self._plot_chipdrag_vs_default_distribution(improvements, viz_dir)
    
    def _plot_hpwl_comparison(self, improvements: List[Dict], viz_dir: Path):
        """绘制HPWL对比图"""
        designs = [i['design'] for i in improvements]
        worst_hpwls = [i.get('worst_hpwl', 0) for i in improvements]
        default_hpwls = [i.get('default_hpwl', 0) for i in improvements]
        optimized_hpwls = [i.get('optimized_hpwl', 0) for i in improvements]
        
        x = range(len(designs))
        width = 0.25
        
        plt.figure(figsize=(15, 8))
        plt.bar([i - width for i in x], worst_hpwls, width, label='极差布局', alpha=0.8)
        plt.bar(x, default_hpwls, width, label='OpenROAD默认', alpha=0.8)
        plt.bar([i + width for i in x], optimized_hpwls, width, label='ChipDRAG优化', alpha=0.8)
        
        plt.xlabel('设计')
        plt.ylabel('HPWL (μm)')
        plt.title('三组布局HPWL对比')
        plt.xticks(x, designs, rotation=45)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        
        plt.savefig(viz_dir / "hpwl_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_comparison(self, improvements: List[Dict], viz_dir: Path):
        """绘制提升率对比图"""
        designs = [i['design'] for i in improvements]
        default_improvements = [i['default_improvement'] for i in improvements]
        optimized_improvements = [i['optimized_improvement'] for i in improvements]
        
        x = range(len(designs))
        width = 0.35
        
        plt.figure(figsize=(15, 8))
        plt.bar([i - width/2 for i in x], default_improvements, width, label='OpenROAD默认提升', alpha=0.8)
        plt.bar([i + width/2 for i in x], optimized_improvements, width, label='ChipDRAG优化提升', alpha=0.8)
        
        plt.xlabel('设计')
        plt.ylabel('提升率 (%)')
        plt.title('HPWL提升率对比')
        plt.xticks(x, designs, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(viz_dir / "improvement_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_chipdrag_vs_default_distribution(self, improvements: List[Dict], viz_dir: Path):
        """绘制ChipDRAG vs 默认提升率分布"""
        chipdrag_vs_default = [i['chipdrag_vs_default'] for i in improvements]
        
        plt.figure(figsize=(10, 6))
        plt.hist(chipdrag_vs_default, bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(chipdrag_vs_default), color='red', linestyle='--', label=f'平均值: {np.mean(chipdrag_vs_default):.2f}%')
        
        plt.xlabel('ChipDRAG vs OpenROAD默认提升率 (%)')
        plt.ylabel('设计数量')
        plt.title('ChipDRAG相对于OpenROAD默认的提升率分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(viz_dir / "chipdrag_vs_default_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """运行完整的论文实验"""
        logger.info("=== 开始论文HPWL对比实验 ===")
        
        # 1. 生成缺失的默认DEF文件
        missing_results = self.generate_missing_default_defs()
        
        # 2. 收集三组HPWL数据
        results = self.collect_three_group_hpwl()
        
        # 3. 生成对比报告
        report = self.generate_comparison_report(results)
        
        # 4. 保存结果
        self.save_results(results, report)
        
        # 5. 生成可视化
        self.generate_visualizations(report)
        
        logger.info("=== 论文HPWL对比实验完成 ===")
        logger.info(f"完成率: {report['summary']['completion_rate']:.2%}")
        logger.info(f"平均ChipDRAG提升: {report['summary']['avg_chipdrag_vs_default']:.2f}%")
        
        return report

def main():
    """主函数"""
    experiment = PaperHPWLComparisonExperiment()
    report = experiment.run_complete_experiment()
    
    # 打印关键结果
    print("\n=== 论文实验关键结果 ===")
    print(f"总设计数: {report['summary']['total_designs']}")
    print(f"完成设计数: {report['summary']['complete_designs']}")
    print(f"完成率: {report['summary']['completion_rate']:.2%}")
    print(f"平均OpenROAD默认提升: {report['summary']['avg_default_improvement']:.2f}%")
    print(f"平均ChipDRAG优化提升: {report['summary']['avg_optimized_improvement']:.2f}%")
    print(f"平均ChipDRAG vs OpenROAD默认: {report['summary']['avg_chipdrag_vs_default']:.2f}%")
    
    return 0

if __name__ == "__main__":
    exit(main()) 