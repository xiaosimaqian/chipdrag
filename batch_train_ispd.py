#!/usr/bin/env python3
"""
ISPD 2015竞赛基准测试批量训练脚本
支持并行处理和详细进度显示
"""

import os
import sys
import time
import json
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入OpenROAD接口
try:
    from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
    OPENROAD_AVAILABLE = True
except ImportError as e:
    logger.error(f"无法导入OpenROAD接口: {e}")
    logger.error("请确保OpenROAD环境已正确配置")
    OPENROAD_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ISPDBatchTrainer:
    """ISPD批量训练器"""
    
    def __init__(self, benchmark_root="data/designs/ispd_2015_contest_benchmark", 
                 results_dir="results/ispd_training", force_retrain=False, max_workers=4):
        """
        初始化批量训练器
        
        Args:
            benchmark_root: ISPD benchmark根目录
            results_dir: 结果保存目录
            force_retrain: 是否强制重新训练（忽略已有结果）
            max_workers: 最大并行工作线程数
        """
        self.benchmark_root = Path(benchmark_root)
        self.results_dir = Path(results_dir)
        self.force_retrain = force_retrain
        self.max_workers = max_workers
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 进度跟踪
        self.completed_designs = 0
        self.total_designs = 0
        self.failed_designs = []
        self.successful_designs = []
        self.lock = threading.Lock()
        
        logger.info(f"批量训练器初始化完成")
        logger.info(f"基准测试目录: {self.benchmark_root}")
        logger.info(f"结果目录: {self.results_dir}")
        logger.info(f"最大并行数: {self.max_workers}")
    
    def _check_docker_resources(self):
        """检查Docker资源是否充足"""
        try:
            # 检查Docker是否运行
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Docker未运行或无法访问")
                return False
            
            # 检查可用内存
            result = subprocess.run(['docker', 'system', 'df'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Docker资源检查通过")
                return True
            return True
        except Exception as e:
            logger.error(f"Docker资源检查失败: {e}")
            return False
    
    def get_design_list(self) -> List[str]:
        """获取所有设计列表"""
        designs = []
        if self.benchmark_root.exists():
            for item in self.benchmark_root.iterdir():
                if item.is_dir() and (item / "design.v").exists():
                    designs.append(item.name)
        designs.sort()
        logger.info(f"发现 {len(designs)} 个设计: {designs}")
        return designs
    
    def update_progress(self, design_name: str, success: bool, message: str = ""):
        """更新进度信息"""
        with self.lock:
            self.completed_designs += 1
            if success:
                self.successful_designs.append(design_name)
            else:
                self.failed_designs.append(design_name)
            
            # 计算进度百分比
            progress = (self.completed_designs / self.total_designs) * 100
            
            logger.info(f"[{self.completed_designs}/{self.total_designs}] {progress:.1f}% - {design_name}: {'成功' if success else '失败'} {message}")
            
            # 实时更新进度条
            if hasattr(self, 'pbar'):
                self.pbar.update(1)
                self.pbar.set_postfix({
                    '成功': len(self.successful_designs),
                    '失败': len(self.failed_designs),
                    '进度': f"{progress:.1f}%"
                })
    
    def train_single_design(self, design_name: str) -> Dict:
        """训练单个设计"""
        # 检查OpenROAD是否可用
        if not OPENROAD_AVAILABLE:
            error_msg = "OpenROAD接口不可用，无法进行真实训练"
            logger.error(f"{design_name}: {error_msg}")
            self.update_progress(design_name, False, error_msg)
            return {"success": False, "error": error_msg}
        
        design_path = os.path.join(self.benchmark_root, design_name)
        
        # 检查结果文件
        result_file = os.path.join(self.results_dir, f"{design_name}_result.json")
        if not self.force_retrain and os.path.exists(result_file):
            logger.info(f"跳过已存在的设计: {design_name}")
            self.update_progress(design_name, True, "已存在，跳过")
            return {"success": True, "skipped": True}
        
        logger.info(f"开始训练设计: {design_name}")
        logger.info(f"设计路径: {design_path}")
        
        try:
            # 初始化OpenROAD接口
            interface = RealOpenROADInterface(design_path)
            
            # 生成TCL脚本
            tcl_script = interface._generate_tcl_script()
            tcl_path = os.path.join(design_path, "openroad_script.tcl")
            
            with open(tcl_path, 'w') as f:
                f.write(tcl_script)
            
            logger.info(f"TCL脚本已生成: {tcl_path}")
            
            # 运行OpenROAD脚本
            start_time = time.time()
            result = interface.run_placement()
            execution_time = time.time() - start_time
            
            # 处理返回的数据格式
            stdout_content = result.get("stdout", [])
            stderr_content = result.get("stderr", [])
            
            # 确保stdout和stderr是字符串
            if isinstance(stdout_content, list):
                stdout_content = "\n".join(stdout_content)
            if isinstance(stderr_content, list):
                stderr_content = "\n".join(stderr_content)
            
            # 保存结果
            result_data = {
                "design_name": design_name,
                "success": result["success"],
                "execution_time": execution_time,
                "wirelength": result.get("metrics", {}).get("wirelength", 0),
                "area": result.get("metrics", {}).get("area", 0),
                "stdout": stdout_content,
                "stderr": stderr_content,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存结果文件
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            # 保存日志文件
            log_file = os.path.join(self.results_dir, f"{design_name}_log.txt")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {design_name} 训练日志 ===\n")
                f.write(f"时间: {datetime.now()}\n")
                f.write(f"成功: {result['success']}\n")
                f.write(f"执行时间: {execution_time:.2f}秒\n\n")
                f.write("=== 标准输出 ===\n")
                f.write(stdout_content)
                f.write("\n\n=== 错误输出 ===\n")
                f.write(stderr_content)
            
            success = result["success"]
            message = f"执行时间: {execution_time:.2f}秒"
            if success:
                message += f", 线长: {result_data['wirelength']:.2f}, 面积: {result_data['area']:.2f}"
            
            self.update_progress(design_name, success, message)
            return result_data
            
        except Exception as e:
            error_msg = f"训练失败: {str(e)}"
            logger.error(f"{design_name}: {error_msg}")
            
            # 保存错误结果
            error_result = {
                "design_name": design_name,
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
            
            self.update_progress(design_name, False, error_msg)
            return error_result
    
    def run_batch_training(self):
        """运行批量训练"""
        logger.info("开始批量训练...")
        
        # 获取设计列表
        designs = self.get_design_list()
        if not designs:
            logger.error("未找到任何设计")
            return
        
        self.total_designs = len(designs)
        logger.info(f"总共 {self.total_designs} 个设计需要训练")
        
        # 检查Docker资源
        if not self._check_docker_resources():
            logger.error("Docker资源检查失败，退出训练")
            return
        
        # 创建进度条
        self.pbar = tqdm(total=self.total_designs, desc="批量训练进度", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        start_time = time.time()
        
        try:
            # 使用线程池进行并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_design = {
                    executor.submit(self.train_single_design, design): design 
                    for design in designs
                }
                
                # 收集结果
                results = []
                for future in concurrent.futures.as_completed(future_to_design):
                    design = future_to_design[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"{design} 执行异常: {e}")
                        self.update_progress(design, False, f"异常: {e}")
        
        finally:
            self.pbar.close()
        
        total_time = time.time() - start_time
        
        # 生成训练总结
        self.generate_training_summary(results, total_time)
        
        logger.info(f"批量训练完成！总时间: {total_time:.2f}秒")
        logger.info(f"成功: {len(self.successful_designs)}, 失败: {len(self.failed_designs)}")
    
    def generate_training_summary(self, results: List[Dict], total_time: float):
        """生成训练总结报告"""
        summary = {
            "training_info": {
                "total_designs": self.total_designs,
                "successful_designs": len(self.successful_designs),
                "failed_designs": len(self.failed_designs),
                "total_time": total_time,
                "average_time": total_time / self.total_designs if self.total_designs > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "successful_designs": self.successful_designs,
            "failed_designs": self.failed_designs,
            "detailed_results": results
        }
        
        # 保存总结报告
        summary_file = self.results_dir / "training_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练总结已保存到: {summary_file}")
        
        # 打印简要统计
        print("\n" + "="*60)
        print("批量训练总结")
        print("="*60)
        print(f"总设计数: {self.total_designs}")
        print(f"成功: {len(self.successful_designs)}")
        print(f"失败: {len(self.failed_designs)}")
        print(f"成功率: {len(self.successful_designs)/self.total_designs*100:.1f}%")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均时间: {total_time/self.total_designs:.2f}秒/设计")
        
        if self.failed_designs:
            print(f"\n失败的设计: {', '.join(self.failed_designs)}")
        
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ISPD 2015竞赛基准测试批量训练")
    parser.add_argument("--benchmark-root", default="data/designs/ispd_2015_contest_benchmark",
                       help="基准测试目录路径")
    parser.add_argument("--results-dir", default="results/ispd_training",
                       help="结果保存目录")
    parser.add_argument("--force-retrain", action="store_true",
                       help="强制重新训练，即使结果已存在")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="最大并行工作线程数")
    
    args = parser.parse_args()
    
    trainer = ISPDBatchTrainer(
        benchmark_root=args.benchmark_root,
        results_dir=args.results_dir,
        force_retrain=args.force_retrain,
        max_workers=args.max_workers
    )
    
    trainer.run_batch_training()

if __name__ == "__main__":
    main() 