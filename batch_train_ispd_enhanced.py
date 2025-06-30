#!/usr/bin/env python3
"""
增强的ISPD 2015批量训练脚本
使用增强的容错机制和错误检查
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_openroad_interface import EnhancedOpenROADInterface
from modules.utils.config_loader import ConfigLoader
from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_batch_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedBatchTrainer:
    """增强的批量训练器，具有更强的容错机制"""
    
    def __init__(self, config_path: str = "configs/experiment_config.json"):
        """初始化增强批量训练器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        config_loader = ConfigLoader()
        self.config = config_loader.load_config("experiment_config.json")
        
        # 设置路径
        self.data_dir = Path("data/designs/ispd_2015_contest_benchmark")
        self.results_dir = Path("results/iterative_training")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化统计
        self.stats = {
            "total_designs": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
            "total_duration": 0
        }
        
        # 初始化OpenROAD接口
        self.openroad_interface = None
        
    def get_ispd_designs(self) -> List[Path]:
        """获取ISPD 2015基准测试设计列表"""
        ispd_dir = Path("data/designs/ispd_2015_contest_benchmark")
        if not ispd_dir.exists():
            logger.error(f"ISPD目录不存在: {ispd_dir}")
            return []
        
        designs = []
        for design_dir in ispd_dir.iterdir():
            if design_dir.is_dir():
                # 检查是否包含必要的文件
                has_lef = any(design_dir.glob("*.lef"))
                has_v = any(design_dir.glob("*.v"))
                has_def = any(design_dir.glob("*.def"))
                
                if has_lef and has_v and has_def:
                    designs.append(design_dir)
                    logger.info(f"找到设计: {design_dir.name}")
                else:
                    logger.warning(f"设计 {design_dir.name} 缺少必要文件 (LEF: {has_lef}, V: {has_v}, DEF: {has_def})")
        
        logger.info(f"总共找到 {len(designs)} 个有效设计")
        return designs
    
    def preprocess_design(self, design_dir: Path) -> bool:
        """预处理设计文件
        
        Args:
            design_dir: 设计目录
            
        Returns:
            bool: 预处理是否成功
        """
        try:
            logger.info(f"预处理设计: {design_dir.name}")
            
            # 1. 检查LEF文件
            lef_files = list(design_dir.glob("*.lef"))
            if not lef_files:
                logger.error(f"设计 {design_dir.name} 没有LEF文件")
                return False
            
            # 2. 检查并修复tech.lef中的SITE定义
            tech_lef = design_dir / "tech.lef"
            if tech_lef.exists():
                # 检查是否包含SITE定义
                with open(tech_lef, 'r') as f:
                    content = f.read()
                    if "SITE core" not in content:
                        logger.info(f"设计 {design_dir.name} 的tech.lef缺少SITE定义，尝试修复")
                        # 从其他LEF文件中提取SITE定义
                        for lef_file in lef_files:
                            if lef_file != tech_lef:
                                with open(lef_file, 'r') as lf:
                                    lef_content = lf.read()
                                    if "SITE core" in lef_content:
                                        # 提取SITE定义并插入到tech.lef开头
                                        site_start = lef_content.find("SITE core")
                                        site_end = lef_content.find("END core", site_start) + 8
                                        site_def = lef_content[site_start:site_end]
                                        
                                        # 备份原文件
                                        shutil.copy2(tech_lef, tech_lef.with_suffix('.lef.backup'))
                                        
                                        # 插入SITE定义
                                        with open(tech_lef, 'r') as f:
                                            original_content = f.read()
                                        
                                        with open(tech_lef, 'w') as f:
                                            f.write(site_def + "\n" + original_content)
                                        
                                        logger.info(f"已修复 {design_dir.name} 的SITE定义")
                                        break
            
            # 3. 检查Verilog文件
            v_files = list(design_dir.glob("*.v"))
            if not v_files:
                logger.error(f"设计 {design_dir.name} 没有Verilog文件")
                return False
            
            # 4. 检查DEF文件
            def_files = list(design_dir.glob("*.def"))
            if not def_files:
                logger.error(f"设计 {design_dir.name} 没有DEF文件")
                return False
            
            logger.info(f"设计 {design_dir.name} 预处理完成")
            return True
            
        except Exception as e:
            logger.error(f"预处理设计 {design_dir.name} 失败: {e}")
            return False
    
    def train_single_design(self, design_dir: Path) -> Dict[str, Any]:
        """训练单个设计
        
        Args:
            design_dir: 设计目录路径
            
        Returns:
            Dict[str, Any]: 训练结果
        """
        design_name = design_dir.name
        
        result = {
            "design": design_name,
            "success": False,
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "output_dir": None
        }
        
        try:
            logger.info(f"开始训练设计: {design_name}")
            
            # 1. 预处理设计
            if not self.preprocess_design(design_dir):
                result["error"] = "预处理失败"
                return result
            
            # 2. 为每个设计创建独立的OpenROAD接口
            openroad_interface = RealOpenROADInterface(work_dir=str(design_dir))
            
            # 3. 运行迭代布局训练
            logger.info(f"开始迭代布局训练，迭代次数: {self.config.get('num_iterations', 10)}")
            
            training_result = openroad_interface.run_iterative_placement(
                num_iterations=self.config.get('num_iterations', 10)
            )
            
            if training_result['success']:
                logger.info(f"✅ 设计 {design_name} 训练成功")
                
                # 保存训练结果
                result['success'] = True
                result['output_dir'] = str(design_dir / "output")
                result['iteration_data'] = training_result.get('iteration_data', [])
                result['execution_time'] = training_result.get('execution_time', 0)
                
                # 提取最佳HPWL
                best_hpwl = float('inf')
                best_iteration = None
                for i, iteration in enumerate(result['iteration_data']):
                    hpwl = iteration.get('hpwl')
                    if hpwl and hpwl != float('inf') and hpwl < best_hpwl:
                        best_hpwl = hpwl
                        best_iteration = i
                
                if best_hpwl != float('inf'):
                    logger.info(f"   最佳HPWL: {best_hpwl:.2e} (轮次 {best_iteration})")
                    result['best_hpwl'] = best_hpwl
                    result['best_iteration'] = best_iteration
                else:
                    logger.warning(f"   未能提取到有效HPWL")
                    result['best_hpwl'] = None
                    result['best_iteration'] = None
                
            else:
                logger.error(f"❌ 设计 {design_name} 训练失败: {training_result.get('error', '未知错误')}")
                result['error'] = training_result.get('error', '训练失败')
                result['success'] = False
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"训练设计 {design_name} 时发生异常: {e}")
        
        finally:
            result["end_time"] = time.time()
            result["duration"] = result["end_time"] - result["start_time"]
        
        return result
    
    def run_batch_training(self, max_workers: int = 4) -> Dict[str, Any]:
        """运行批量训练
        
        Args:
            max_workers: 最大并发工作线程数
            
        Returns:
            Dict[str, Any]: 批量训练结果
        """
        logger.info("开始增强批量训练")
        self.stats["start_time"] = time.time()
        
        # 获取设计列表
        designs = self.get_ispd_designs()
        if not designs:
            logger.error("没有找到可用的设计")
            return {"success": False, "error": "没有找到可用的设计"}
        
        self.stats["total_designs"] = len(designs)
        logger.info(f"准备训练 {len(designs)} 个设计")
        
        # 创建结果目录
        results_file = self.results_dir / "batch_training_results.json"
        
        # 使用线程池进行并发训练
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_design = {
                executor.submit(self.train_single_design, design): design 
                for design in designs
            }
            
            # 收集结果
            for future in as_completed(future_to_design):
                design = future_to_design[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新统计
                    if result["success"]:
                        self.stats["successful"] += 1
                        logger.info(f"✅ 设计 {design.name} 训练成功")
                    else:
                        self.stats["failed"] += 1
                        logger.error(f"❌ 设计 {design.name} 训练失败: {result['error']}")
                    
                    # 保存中间结果
                    with open(results_file, 'w') as f:
                        json.dump({
                            "stats": self.stats,
                            "results": results
                        }, f, indent=2, default=str)
                    
                except Exception as e:
                    logger.error(f"处理设计 {design.name} 结果时发生异常: {e}")
                    self.stats["failed"] += 1
        
        # 完成统计
        self.stats["end_time"] = time.time()
        self.stats["total_duration"] = self.stats["end_time"] - self.stats["start_time"]
        
        # 保存最终结果
        final_results = {
            "stats": self.stats,
            "results": results,
            "summary": {
                "success_rate": self.stats["successful"] / self.stats["total_designs"] if self.stats["total_designs"] > 0 else 0,
                "average_duration": sum(r["duration"] for r in results) / len(results) if results else 0
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # 生成训练报告
        self.generate_training_report(final_results)
        
        logger.info("批量训练完成")
        logger.info(f"成功: {self.stats['successful']}/{self.stats['total_designs']}")
        logger.info(f"失败: {self.stats['failed']}/{self.stats['total_designs']}")
        logger.info(f"成功率: {final_results['summary']['success_rate']:.2%}")
        
        return final_results
    
    def generate_training_report(self, results: Dict[str, Any]):
        """生成训练报告
        
        Args:
            results: 训练结果
        """
        report_file = self.results_dir / "training_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 增强批量训练报告\n\n")
            
            # 统计信息
            stats = results["stats"]
            f.write("## 训练统计\n\n")
            f.write(f"- **总设计数**: {stats['total_designs']}\n")
            f.write(f"- **成功**: {stats['successful']}\n")
            f.write(f"- **失败**: {stats['failed']}\n")
            f.write(f"- **成功率**: {results['summary']['success_rate']:.2%}\n")
            f.write(f"- **平均耗时**: {results['summary']['average_duration']:.2f}秒\n")
            f.write(f"- **总耗时**: {stats['total_duration']:.2f}秒\n\n")
            
            # 成功的设计
            successful_designs = [r for r in results["results"] if r["success"]]
            if successful_designs:
                f.write("## 成功的设计\n\n")
                for design in successful_designs:
                    f.write(f"- {design['design']} (耗时: {design['duration']:.2f}秒)\n")
                f.write("\n")
            
            # 失败的设计
            failed_designs = [r for r in results["results"] if not r["success"]]
            if failed_designs:
                f.write("## 失败的设计\n\n")
                for design in failed_designs:
                    f.write(f"- {design['design']}: {design['error']}\n")
                f.write("\n")
            
            # 详细结果
            f.write("## 详细结果\n\n")
            for result in results["results"]:
                f.write(f"### {result['design']}\n\n")
                f.write(f"- **状态**: {'✅ 成功' if result['success'] else '❌ 失败'}\n")
                f.write(f"- **耗时**: {result['duration']:.2f}秒\n")
                if result['error']:
                    f.write(f"- **错误**: {result['error']}\n")
                if result['output_dir']:
                    f.write(f"- **输出目录**: {result['output_dir']}\n")
                f.write("\n")
        
        logger.info(f"训练报告已生成: {report_file}")

def main():
    """主函数"""
    # 创建增强批量训练器
    trainer = EnhancedBatchTrainer()
    
    # 运行批量训练
    results = trainer.run_batch_training(max_workers=2)  # 使用2个并发线程
    
    if results["stats"]["successful"] > 0:
        logger.info("✅ 批量训练完成，有成功的设计")
        return 0
    else:
        logger.error("❌ 批量训练失败，没有成功的设计")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 