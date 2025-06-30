#!/usr/bin/env python3
"""
ISPD 2015基准测试批量训练脚本
支持迭代布局模式，生成RL训练数据
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchTrainer:
    """批量训练器"""
    
    def __init__(self, benchmark_dir: str, results_dir: str, use_iterative: bool = False, num_iterations: int = 10):
        self.benchmark_dir = Path(benchmark_dir)
        self.results_dir = Path(results_dir)
        self.use_iterative = use_iterative
        self.num_iterations = num_iterations
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设计规模分类
        self.large_designs = [
            'mgc_superblue16_a', 'mgc_superblue11_a', 'mgc_superblue12',
            'mgc_des_perf_a', 'mgc_des_perf_1', 'mgc_des_perf_b'
        ]
        self.medium_designs = [
            'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b',
            'mgc_matrix_mult_a', 'mgc_matrix_mult_b', 'mgc_matrix_mult_1'
        ]
        self.small_designs = [
            'mgc_fft_a', 'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_b',
            'mgc_edit_dist_a'
        ]
        
        logger.info("批量训练器初始化完成")
        logger.info(f"基准测试目录: {self.benchmark_dir}")
        logger.info(f"结果目录: {self.results_dir}")
        if self.use_iterative:
            logger.info(f"使用迭代布局模式，迭代次数: {self.num_iterations}")
    
    def get_design_directories(self) -> List[Path]:
        """获取所有设计目录"""
        design_dirs = []
        for item in self.benchmark_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                design_dirs.append(item)
        return sorted(design_dirs)
    
    def classify_designs(self, design_dirs: List[Path]) -> Dict[str, List[Path]]:
        """按规模分类设计"""
        classified = {
            'large': [],
            'medium': [],
            'small': []
        }
        
        for design_dir in design_dirs:
            design_name = design_dir.name
            if design_name in self.large_designs:
                classified['large'].append(design_dir)
            elif design_name in self.medium_designs:
                classified['medium'].append(design_dir)
            elif design_name in self.small_designs:
                classified['small'].append(design_dir)
            else:
                # 默认归类为中等规模
                classified['medium'].append(design_dir)
        
        return classified
    
    def train_single_design(self, design_dir: Path) -> Dict[str, Any]:
        """训练单个设计"""
        design_name = design_dir.name
        logger.info(f"开始训练设计: {design_name}")
        
        try:
            # 创建OpenROAD接口
            interface = RealOpenROADInterface(str(design_dir))
            
            if self.use_iterative:
                # 迭代布局模式
                logger.info(f"使用迭代布局模式，迭代次数: {self.num_iterations}")
                result = interface.run_iterative_placement(self.num_iterations)
                
                if result.get("success", False):
                    # 收集迭代数据
                    iteration_data = interface.collect_iteration_data()
                    result["iteration_data"] = iteration_data
                    logger.info(f"设计 {design_name} 迭代训练成功，生成 {len(iteration_data)} 个迭代结果")
                else:
                    logger.error(f"设计 {design_name} 迭代训练失败: {result.get('error', 'Unknown error')}")
            else:
                # 单次布局模式
                result = interface.run_placement()
                
                if result.get("success", False):
                    logger.info(f"设计 {design_name} 训练成功")
                else:
                    logger.error(f"设计 {design_name} 训练失败: {result.get('error', 'Unknown error')}")
            
            # 添加设计信息
            result["design_name"] = design_name
            result["design_dir"] = str(design_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"设计 {design_name} 训练异常: {e}")
            return {
                "design_name": design_name,
                "design_dir": str(design_dir),
                "success": False,
                "error": str(e)
            }
    
    def train_design_batch(self, design_dirs: List[Path], max_workers: int = 4) -> List[Dict[str, Any]]:
        """批量训练设计"""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_design = {
                executor.submit(self.train_single_design, design_dir): design_dir 
                for design_dir in design_dirs
            }
            
            # 收集结果
            for future in as_completed(future_to_design):
                design_dir = future_to_design[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 保存单个设计结果
                    result_file = self.results_dir / f"{design_dir.name}_result.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"设计 {design_dir.name} 训练完成，结果保存到: {result_file}")
                    
                except Exception as e:
                    logger.error(f"设计 {design_dir.name} 训练失败: {e}")
                    error_result = {
                        "design_name": design_dir.name,
                        "design_dir": str(design_dir),
                        "success": False,
                        "error": str(e)
                    }
                    results.append(error_result)
                    
                    # 保存错误结果
                    result_file = self.results_dir / f"{design_dir.name}_result.json"
                    with open(result_file, 'w') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)
        
        return results
    
    def run_batch_training(self, max_workers: int = 4):
        """运行批量训练"""
        # 获取所有设计目录
        design_dirs = self.get_design_directories()
        logger.info(f"找到 {len(design_dirs)} 个设计目录")
        
        # 按规模分类设计
        classified_designs = self.classify_designs(design_dirs)
        
        logger.info(f"设计分类: 大规模={len(classified_designs['large'])}, 中等规模={len(classified_designs['medium'])}, 小规模={len(classified_designs['small'])}")
        
        all_results = []
        
        # 分批处理：先处理小规模设计，再处理中等规模，最后处理大规模设计
        for size, designs in [('small', classified_designs['small']), 
                             ('medium', classified_designs['medium']), 
                             ('large', classified_designs['large'])]:
            if designs:
                logger.info(f"开始处理{size}规模设计，数量: {len(designs)}")
                
                # 根据规模调整并发数
                if size == 'large':
                    workers = min(2, max_workers)  # 大规模设计减少并发
                elif size == 'medium':
                    workers = min(3, max_workers)  # 中等规模设计中等并发
                else:
                    workers = max_workers  # 小规模设计可以高并发
                
                batch_results = self.train_design_batch(designs, max_workers=workers)
                all_results.extend(batch_results)
                
                logger.info(f"{size}规模设计处理完成")
        
        # 生成训练报告
        self.generate_training_report(all_results)
        
        logger.info("批量训练完成")
        return all_results
    
    def generate_training_report(self, results: List[Dict[str, Any]]):
        """生成训练报告"""
        report_file = self.results_dir / "iterative_training_report.md"
        
        # 统计结果
        total_designs = len(results)
        successful_designs = sum(1 for r in results if r.get("success", False))
        failed_designs = total_designs - successful_designs
        
        # 按规模统计
        size_stats = {"large": 0, "medium": 0, "small": 0}
        success_by_size = {"large": 0, "medium": 0, "small": 0}
        
        for result in results:
            design_name = result["design_name"]
            if design_name in self.large_designs:
                size_stats["large"] += 1
                if result.get("success", False):
                    success_by_size["large"] += 1
            elif design_name in self.medium_designs:
                size_stats["medium"] += 1
                if result.get("success", False):
                    success_by_size["medium"] += 1
            elif design_name in self.small_designs:
                size_stats["small"] += 1
                if result.get("success", False):
                    success_by_size["small"] += 1
        
        # 生成报告内容
        report_content = f"""# ISPD 2015 迭代布局训练报告

## 训练概览
- **总设计数量**: {total_designs}
- **成功设计数量**: {successful_designs}
- **失败设计数量**: {failed_designs}
- **成功率**: {successful_designs/total_designs*100:.1f}%

## 按规模统计
| 规模 | 总数 | 成功 | 成功率 |
|------|------|------|--------|
| 大规模 | {size_stats["large"]} | {success_by_size["large"]} | {success_by_size["large"]/max(size_stats["large"], 1)*100:.1f}% |
| 中等规模 | {size_stats["medium"]} | {success_by_size["medium"]} | {success_by_size["medium"]/max(size_stats["medium"], 1)*100:.1f}% |
| 小规模 | {size_stats["small"]} | {success_by_size["small"]} | {success_by_size["small"]/max(size_stats["small"], 1)*100:.1f}% |

## 训练配置
- **迭代模式**: {'是' if self.use_iterative else '否'}
- **迭代次数**: {self.num_iterations if self.use_iterative else 'N/A'}
- **Docker资源限制**: 8GB内存, 4CPU核心

## 详细结果
"""
        
        # 添加每个设计的详细结果
        for result in results:
            design_name = result["design_name"]
            success = result.get("success", False)
            error = result.get("error", "")
            
            report_content += f"""
### {design_name}
- **状态**: {'成功' if success else '失败'}
"""
            
            if not success and error:
                report_content += f"- **错误**: {error}\n"
            
            if success and self.use_iterative:
                iteration_data = result.get("iteration_data", [])
                report_content += f"- **迭代数据**: {len(iteration_data)} 个迭代结果\n"
        
        # 写入报告文件
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"迭代训练报告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="ISPD 2015基准测试批量训练")
    parser.add_argument("--benchmark-dir", default="data/designs/ispd_2015_contest_benchmark", 
                       help="基准测试目录")
    parser.add_argument("--results-dir", default="results/batch_training", 
                       help="结果保存目录")
    parser.add_argument("--use-iterative", action="store_true", 
                       help="使用迭代布局模式")
    parser.add_argument("--num-iterations", type=int, default=10, 
                       help="迭代次数（仅在迭代模式下有效）")
    parser.add_argument("--max-workers", type=int, default=4, 
                       help="最大并发工作进程数")
    
    args = parser.parse_args()
    
    # 创建批量训练器
    trainer = BatchTrainer(
        benchmark_dir=args.benchmark_dir,
        results_dir=args.results_dir,
        use_iterative=args.use_iterative,
        num_iterations=args.num_iterations
    )
    
    # 运行批量训练
    if args.use_iterative:
        logger.info(f"开始迭代布局训练，设计数量: {len(trainer.get_design_directories())}, 迭代次数: {args.num_iterations}")
    else:
        logger.info(f"开始单次布局训练，设计数量: {len(trainer.get_design_directories())}")
    
    results = trainer.run_batch_training(max_workers=args.max_workers)
    
    # 输出统计信息
    successful = sum(1 for r in results if r.get("success", False))
    logger.info(f"训练完成: {successful}/{len(results)} 个设计成功")

if __name__ == "__main__":
    main() 