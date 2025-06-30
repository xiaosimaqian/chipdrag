#!/usr/bin/env python3
"""
补全HPWL数据脚本
从已有的训练结果中提取HPWL并补全到batch_training_results.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPWLDataCompleter:
    """HPWL数据补全器"""
    
    def __init__(self, results_dir: str = "results/iterative_training"):
        """
        初始化补全器
        
        Args:
            results_dir: 训练结果目录
        """
        self.results_dir = Path(results_dir)
        self.training_file = self.results_dir / "batch_training_results.json"
        self.output_file = self.results_dir / "batch_training_results_with_hpwl.json"
        
    def _extract_hpwl_from_log(self, log_file: Path) -> Optional[float]:
        """
        从日志文件中提取HPWL值
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            float: HPWL值，如果提取失败返回None
        """
        try:
            if not log_file.exists():
                return None
            
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找HPWL相关的行
            lines = content.split('\n')
            hpwl_values = []
            
            for line in lines:
                # 匹配InitialPlace中的HPWL格式: "HPWL: 3165291948"
                if '[InitialPlace]' in line and 'HPWL:' in line:
                    # 提取数值
                    numbers = re.findall(r'HPWL:\s*(\d+)', line)
                    if numbers:
                        try:
                            hpwl_values.append(float(numbers[0]))
                        except ValueError:
                            continue
                
                # 匹配其他HPWL格式
                elif any(keyword in line for keyword in ['HPWL:', 'Wirelength:', 'Total wirelength:', 'HPWL =']):
                    # 提取数值
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        try:
                            hpwl_values.append(float(numbers[0]))
                        except ValueError:
                            continue
            
            # 返回最后一个HPWL值（通常是最终结果）
            if hpwl_values:
                return hpwl_values[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"从日志提取HPWL时发生异常: {e}")
            return None
    
    def _extract_hpwl_from_def(self, def_file: Path) -> Optional[float]:
        """
        从DEF文件中提取HPWL值（通过OpenROAD命令）
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            float: HPWL值，如果提取失败返回None
        """
        try:
            if not def_file.exists():
                return None
            
            # 使用OpenROAD命令提取HPWL
            import subprocess
            work_dir = def_file.parent
            def_name = def_file.name
            
            cmd = f"docker run --rm -v {work_dir}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit - << EOF\nread_def {def_name}\nreport_wirelength\nEOF'"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # 解析输出中的HPWL值
                for line in result.stdout.split('\n'):
                    if 'HPWL' in line and ':' in line:
                        # 提取数值
                        parts = line.split(':')
                        if len(parts) >= 2:
                            hpwl_str = parts[1].strip().split()[0]  # 取第一个数值
                            try:
                                return float(hpwl_str)
                            except ValueError:
                                continue
            
            return None
            
        except Exception as e:
            logger.error(f"从DEF文件提取HPWL时发生异常: {e}")
            return None
    
    def _find_hpwl_in_design_dir(self, design_dir: Path) -> List[Dict[str, Any]]:
        """
        在设计目录中查找HPWL数据
        
        Args:
            design_dir: 设计目录路径
            
        Returns:
            List[Dict]: 包含HPWL的迭代数据列表
        """
        iteration_data = []
        
        # 查找可能的日志文件
        log_files = [
            design_dir / "placement_iterations.log",
            design_dir / "openroad_execution.log",
            design_dir / "output" / "placement_iterations.log"
        ]
        
        # 查找可能的DEF文件
        def_files = [
            design_dir / "placement_result.def",
            design_dir / "output" / "placement_result.def"
        ]
        
        # 尝试从日志文件提取HPWL
        hpwl_from_log = None
        for log_file in log_files:
            if log_file.exists():
                hpwl_from_log = self._extract_hpwl_from_log(log_file)
                if hpwl_from_log is not None:
                    logger.info(f"从日志文件提取HPWL: {hpwl_from_log:.2e}")
                    break
        
        # 尝试从DEF文件提取HPWL
        hpwl_from_def = None
        for def_file in def_files:
            if def_file.exists():
                hpwl_from_def = self._extract_hpwl_from_def(def_file)
                if hpwl_from_def is not None:
                    logger.info(f"从DEF文件提取HPWL: {hpwl_from_def:.2e}")
                    break
        
        # 优先使用DEF文件的HPWL，如果都没有则使用日志的
        final_hpwl = hpwl_from_def if hpwl_from_def is not None else hpwl_from_log
        
        if final_hpwl is not None:
            # 创建迭代数据
            iteration_data.append({
                "iteration": 0,
                "hpwl": final_hpwl,
                "source": "def" if hpwl_from_def is not None else "log"
            })
        
        return iteration_data
    
    def complete_hpwl_data(self) -> bool:
        """
        补全HPWL数据
        
        Returns:
            bool: 是否成功补全
        """
        try:
            # 读取原始训练结果
            if not self.training_file.exists():
                logger.error(f"训练结果文件不存在: {self.training_file}")
                return False
            
            with open(self.training_file, 'r') as f:
                training_data = json.load(f)
            
            logger.info(f"开始补全HPWL数据，共 {len(training_data.get('results', []))} 个设计")
            
            # 遍历每个设计
            completed_count = 0
            for result in training_data.get('results', []):
                design_name = result.get('design')
                if not design_name:
                    continue
                
                logger.info(f"处理设计: {design_name}")
                
                # 检查是否已有HPWL数据
                has_hpwl = False
                if 'iteration_data' in result:
                    for iteration in result['iteration_data']:
                        if iteration.get('hpwl') is not None and iteration['hpwl'] != float('inf'):
                            has_hpwl = True
                            break
                
                if has_hpwl:
                    logger.info(f"  设计 {design_name} 已有HPWL数据，跳过")
                    continue
                
                # 查找设计目录
                design_dir = Path("data/designs/ispd_2015_contest_benchmark") / design_name
                if not design_dir.exists():
                    logger.warning(f"  设计目录不存在: {design_dir}")
                    continue
                
                # 提取HPWL数据
                hpwl_data = self._find_hpwl_in_design_dir(design_dir)
                
                if hpwl_data:
                    # 补全到结果中
                    if 'iteration_data' not in result:
                        result['iteration_data'] = []
                    
                    # 合并HPWL数据
                    for hpwl_item in hpwl_data:
                        iteration_num = hpwl_item['iteration']
                        # 查找对应的迭代数据
                        found = False
                        for iteration in result['iteration_data']:
                            if iteration.get('iteration') == iteration_num:
                                iteration['hpwl'] = hpwl_item['hpwl']
                                found = True
                                break
                        
                        if not found:
                            # 如果没有对应的迭代数据，创建一个
                            result['iteration_data'].append({
                                'iteration': iteration_num,
                                'hpwl': hpwl_item['hpwl']
                            })
                    
                    completed_count += 1
                    logger.info(f"  ✅ 设计 {design_name} HPWL数据补全成功")
                else:
                    logger.warning(f"  ❌ 设计 {design_name} 无法提取HPWL数据")
            
            # 保存补全后的结果
            with open(self.output_file, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)
            
            logger.info(f"HPWL数据补全完成！")
            logger.info(f"  成功补全: {completed_count} 个设计")
            logger.info(f"  输出文件: {self.output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"补全HPWL数据时发生异常: {e}")
            return False

def main():
    """主函数"""
    # 创建补全器
    completer = HPWLDataCompleter()
    
    # 执行补全
    success = completer.complete_hpwl_data()
    
    if success:
        logger.info("HPWL数据补全成功！")
    else:
        logger.error("HPWL数据补全失败！")

if __name__ == "__main__":
    main() 