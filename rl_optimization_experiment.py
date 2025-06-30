#!/usr/bin/env python3
"""
强化学习优化验证实验
实验1: 同设计优化验证 - 验证检索参数是否能优化同一设计的HPWL
实验2: 跨设计泛化验证 - 验证检索参数是否能泛化到相似规模的设计
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from modules.rl_training.real_openroad_interface_fixed import RealOpenROADInterface
from modules.utils.config_loader import ConfigLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DesignInfo:
    """设计信息"""
    name: str
    num_instances: int
    num_nets: int
    area: Tuple[int, int]
    baseline_hpwl: Optional[float] = None
    baseline_params: Optional[Dict] = None

@dataclass
class OptimizationResult:
    """优化结果"""
    design_name: str
    experiment_type: str  # "same_design" or "cross_design"
    source_design: str
    target_design: str
    retrieved_params: Dict[str, Any]
    optimized_hpwl: Optional[float]
    baseline_hpwl: Optional[float]
    improvement: float  # 百分比改进
    execution_time: float
    success: bool
    error: Optional[str] = None

class RLOptimizationExperiment:
    """强化学习优化验证实验"""
    
    def __init__(self, results_dir: str = "results/iterative_training"):
        """
        初始化实验
        
        Args:
            results_dir: 训练结果目录
        """
        self.results_dir = Path(results_dir)
        self.data_dir = Path("data/designs/ispd_2015_contest_benchmark")
        self.experiment_results_dir = Path("results/rl_optimization_experiment")
        self.experiment_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        config_loader = ConfigLoader()
        self.config = config_loader.load_config("experiment_config.json")
        
        # 加载训练数据
        # 优先读取带HPWL的补全文件
        hpwl_file = self.results_dir / "batch_training_results_with_hpwl.json"
        if hpwl_file.exists():
            with open(hpwl_file, 'r') as f:
                self.training_data = json.load(f)
            logger.info(f"已加载补全HPWL的训练数据: {hpwl_file}")
        else:
            self.training_data = self._load_training_data()
        
        # 设计信息映射
        self.design_info = self._build_design_info()
        
        logger.info(f"实验初始化完成，找到 {len(self.design_info)} 个设计")
    
    def _load_training_data(self) -> Dict[str, Any]:
        """加载训练数据"""
        training_file = self.results_dir / "batch_training_results.json"
        if training_file.exists():
            with open(training_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"训练结果文件不存在: {training_file}")
            return {}
    
    def _build_design_info(self) -> Dict[str, DesignInfo]:
        """构建设计信息字典"""
        design_info = {}
        
        # 训练数据是列表结构
        for design_data in self.training_data['results']:
            design_name = design_data['design']
            if not design_data.get('success', False):
                continue
                
            # 提取设计统计信息
            num_instances = design_data.get('num_instances', 0)
            num_nets = design_data.get('num_nets', 0)
            area = design_data.get('area', (0, 0))
            
            # 提取基线HPWL - 使用迭代0的HPWL作为基线
            baseline_hpwl = None
            iteration_data = design_data.get('iteration_data', [])
            
            for iteration in iteration_data:
                if iteration.get('iteration') == 0:
                    hpwl = iteration.get('hpwl')
                    if hpwl is not None:
                        # 检查HPWL值的合理性
                        if hpwl > 1000:  # 至少1000才可能是合理的HPWL
                            baseline_hpwl = hpwl
                        else:
                            print(f"警告: {design_name} 的基线HPWL值异常小 ({hpwl})，将被忽略")
                    break
            
            # 提取基线参数
            baseline_params = design_data.get('baseline_params', {})
            
            design_info[design_name] = DesignInfo(
                name=design_name,
                num_instances=num_instances,
                num_nets=num_nets,
                area=area,
                baseline_hpwl=baseline_hpwl,
                baseline_params=baseline_params
            )
        
        print(f"构建了 {len(design_info)} 个设计的信息")
        print("有效的基线HPWL设计:")
        for name, info in design_info.items():
            if info.baseline_hpwl and info.baseline_hpwl > 1000:
                print(f"  {name}: {info.baseline_hpwl:.2e}")
        
        return design_info
    
    def _retrieve_optimization_params(self, target_design: str, source_design: str = None) -> Dict[str, Any]:
        """
        检索优化参数
        
        Args:
            target_design: 目标设计
            source_design: 源设计（用于跨设计实验）
            
        Returns:
            检索到的优化参数
        """
        target_info = self.design_info.get(target_design)
        if not target_info:
            return self._get_default_params()
        
        if source_design:
            # 跨设计检索：使用源设计的参数
            source_info = self.design_info.get(source_design)
            if source_info and source_info.baseline_params:
                # 根据设计规模调整参数
                scale_factor = target_info.num_instances / source_info.num_instances
                
                retrieved_params = source_info.baseline_params.copy()
                # 根据规模调整密度
                if scale_factor > 1.5:
                    retrieved_params['density'] = max(0.6, retrieved_params['density'] - 0.05)
                elif scale_factor < 0.7:
                    retrieved_params['density'] = min(0.85, retrieved_params['density'] + 0.05)
                
                return retrieved_params
        else:
            # 同设计检索：使用历史最佳参数
            if target_info.baseline_params:
                return target_info.baseline_params.copy()
        
        return self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """获取默认参数"""
        return {
            'density': 0.75,
            'wirelength_coef': 1.0,
            'density_penalty': 0.0001,
            'max_displacement': 5,
            'max_iterations': 5
        }
    
    def _run_optimization(self, design_name: str, params: Dict[str, Any]) -> OptimizationResult:
        """
        运行优化
        
        Args:
            design_name: 设计名称
            params: 优化参数
            
        Returns:
            优化结果
        """
        design_dir = self.data_dir / design_name
        if not design_dir.exists():
            return OptimizationResult(
                design_name=design_name,
                experiment_type="unknown",
                source_design="",
                target_design=design_name,
                retrieved_params=params,
                optimized_hpwl=None,
                baseline_hpwl=None,
                improvement=0.0,
                execution_time=0.0,
                success=False,
                error="设计目录不存在"
            )
        
        try:
            logger.info(f"开始优化设计 {design_name}，参数: {params}")
            
            # 创建OpenROAD接口
            interface = RealOpenROADInterface(work_dir=str(design_dir))
            
            # 运行优化
            start_time = time.time()
            result = interface.run_placement(
                density_target=params.get('density', 0.75),
                wirelength_weight=params.get('wirelength_coef', 1.0),
                density_weight=params.get('density_penalty', 0.0001)
            )
            execution_time = time.time() - start_time
            
            if result['success']:
                # 提取HPWL
                optimized_hpwl = result.get('hpwl', result.get('wirelength', None))
                baseline_hpwl = self.design_info.get(design_name, DesignInfo("", 0, 0, (0, 0))).baseline_hpwl
                
                if baseline_hpwl is not None and optimized_hpwl is not None:
                    improvement = ((baseline_hpwl - optimized_hpwl) / baseline_hpwl) * 100
                else:
                    improvement = 0.0
                
                return OptimizationResult(
                    design_name=design_name,
                    experiment_type="unknown",
                    source_design="",
                    target_design=design_name,
                    retrieved_params=params,
                    optimized_hpwl=optimized_hpwl,
                    baseline_hpwl=baseline_hpwl,
                    improvement=improvement,
                    execution_time=execution_time,
                    success=True
                )
            else:
                return OptimizationResult(
                    design_name=design_name,
                    experiment_type="unknown",
                    source_design="",
                    target_design=design_name,
                    retrieved_params=params,
                    optimized_hpwl=None,
                    baseline_hpwl=None,
                    improvement=0.0,
                    execution_time=execution_time,
                    success=False,
                    error=result.get('stderr', '未知错误')
                )
                
        except Exception as e:
            logger.error(f"优化设计 {design_name} 时发生异常: {e}")
            return OptimizationResult(
                design_name=design_name,
                experiment_type="unknown",
                source_design="",
                target_design=design_name,
                retrieved_params=params,
                optimized_hpwl=None,
                baseline_hpwl=None,
                improvement=0.0,
                execution_time=0.0,
                success=False,
                error=str(e)
            )
    
    def experiment_1_same_design_optimization(self) -> List[OptimizationResult]:
        """
        实验1: 同设计优化验证
        
        Returns:
            优化结果列表
        """
        logger.info("开始实验1: 同设计优化验证")
        
        results = []
        # 只选择有有效基线HPWL的设计
        successful_designs = [name for name, info in self.design_info.items() 
                            if info.baseline_hpwl and info.baseline_hpwl > 1000]
        
        logger.info(f"找到 {len(successful_designs)} 个有有效基线HPWL的设计")
        logger.info(f"有效设计: {successful_designs}")
        
        if not successful_designs:
            logger.warning("没有找到有效的设计，跳过实验1")
            return results
        
        for design_name in successful_designs:
            logger.info(f"优化设计: {design_name}")
            
            # 检索优化参数
            params = self._retrieve_optimization_params(design_name)
            
            # 运行优化
            result = self._run_optimization(design_name, params)
            result.experiment_type = "same_design"
            result.source_design = design_name
            
            results.append(result)
            
            # 输出结果
            if result.success:
                logger.info(f"✅ {design_name}: HPWL从 {result.baseline_hpwl:.2e} 优化到 {result.optimized_hpwl:.2e} "
                          f"(改进: {result.improvement:.2f}%)")
            else:
                logger.error(f"❌ {design_name}: 优化失败 - {result.error}")
        
        return results
    
    def experiment_2_cross_design_generalization(self) -> List[OptimizationResult]:
        """
        实验2: 跨设计泛化验证
        
        Returns:
            优化结果列表
        """
        logger.info("开始实验2: 跨设计泛化验证")
        
        results = []
        
        # 只选择有有效基线HPWL的设计
        valid_designs = [name for name, info in self.design_info.items() 
                        if info.baseline_hpwl and info.baseline_hpwl > 1000]
        
        logger.info(f"找到 {len(valid_designs)} 个有有效基线HPWL的设计用于跨设计实验")
        
        if len(valid_designs) < 2:
            logger.warning("有效设计数量不足，无法进行跨设计实验")
            return results
        
        # 按设计规模分组
        small_designs = []
        medium_designs = []
        large_designs = []
        
        for name in valid_designs:
            info = self.design_info[name]
            if info.num_instances < 50000:
                small_designs.append(name)
            elif info.num_instances < 200000:
                medium_designs.append(name)
            else:
                large_designs.append(name)
        
        logger.info(f"设计分组: 小型({len(small_designs)}), 中型({len(medium_designs)}), 大型({len(large_designs)})")
        
        # 在相似规模的设计间进行交叉验证
        design_groups = [small_designs, medium_designs, large_designs]
        
        for group_idx, group in enumerate(design_groups):
            if len(group) < 2:
                logger.info(f"第{group_idx+1}组设计数量不足({len(group)})，跳过")
                continue
                
            logger.info(f"处理第{group_idx+1}组设计: {group}")
            
            for i, source_design in enumerate(group):
                for j, target_design in enumerate(group):
                    if i != j:  # 避免自己优化自己
                        logger.info(f"用 {source_design} 的经验优化 {target_design}")
                        
                        # 检索优化参数
                        params = self._retrieve_optimization_params(target_design, source_design)
                        
                        # 运行优化
                        result = self._run_optimization(target_design, params)
                        result.experiment_type = "cross_design"
                        result.source_design = source_design
                        
                        results.append(result)
                        
                        # 输出结果
                        if result.success:
                            logger.info(f"✅ {source_design}→{target_design}: HPWL从 {result.baseline_hpwl:.2e} "
                                      f"优化到 {result.optimized_hpwl:.2e} (改进: {result.improvement:.2f}%)")
                        else:
                            logger.error(f"❌ {source_design}→{target_design}: 优化失败 - {result.error}")
        
        return results
    
    def run_all_experiments(self) -> Dict[str, List[OptimizationResult]]:
        """
        运行所有实验
        
        Returns:
            实验结果字典
        """
        logger.info("开始运行强化学习优化验证实验")
        
        # 实验1: 同设计优化验证
        experiment_1_results = self.experiment_1_same_design_optimization()
        
        # 实验2: 跨设计泛化验证
        experiment_2_results = self.experiment_2_cross_design_generalization()
        
        # 保存结果
        all_results = {
            "experiment_1_same_design": experiment_1_results,
            "experiment_2_cross_design": experiment_2_results
        }
        
        self._save_results(all_results)
        self._generate_report(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, List[OptimizationResult]]):
        """保存实验结果"""
        results_file = self.experiment_results_dir / "optimization_results.json"
        
        # 自定义JSON编码器，处理特殊值
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
                    return None
                return super().default(obj)
        
        # 转换为可序列化的格式
        serializable_results = {}
        for exp_name, exp_results in results.items():
            serializable_results[exp_name] = []
            for result in exp_results:
                # 处理HPWL值，将None或inf转换为null
                optimized_hpwl = result.optimized_hpwl
                baseline_hpwl = result.baseline_hpwl
                
                if optimized_hpwl is None or (isinstance(optimized_hpwl, float) and (optimized_hpwl == float('inf') or optimized_hpwl == float('-inf'))):
                    optimized_hpwl = None
                
                if baseline_hpwl is None or (isinstance(baseline_hpwl, float) and (baseline_hpwl == float('inf') or baseline_hpwl == float('-inf'))):
                    baseline_hpwl = None
                
                serializable_results[exp_name].append({
                    'design_name': result.design_name,
                    'experiment_type': result.experiment_type,
                    'source_design': result.source_design,
                    'target_design': result.target_design,
                    'retrieved_params': result.retrieved_params,
                    'optimized_hpwl': optimized_hpwl,
                    'baseline_hpwl': baseline_hpwl,
                    'improvement': result.improvement,
                    'execution_time': result.execution_time,
                    'success': result.success,
                    'error': result.error
                })
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=CustomJSONEncoder)
        
        logger.info(f"实验结果已保存到: {results_file}")
    
    def _generate_report(self, results: Dict[str, List[OptimizationResult]]):
        """生成实验报告"""
        report_file = self.experiment_results_dir / "optimization_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 强化学习优化验证实验报告\n\n")
            
            # 实验1报告
            f.write("## 实验1: 同设计优化验证\n\n")
            exp1_results = results["experiment_1_same_design"]
            
            successful_exp1 = [r for r in exp1_results if r.success]
            failed_exp1 = [r for r in exp1_results if not r.success]
            
            f.write(f"- **总实验数**: {len(exp1_results)}\n")
            f.write(f"- **成功数**: {len(successful_exp1)}\n")
            f.write(f"- **失败数**: {len(failed_exp1)}\n")
            
            if successful_exp1:
                improvements = [r.improvement for r in successful_exp1]
                f.write(f"- **平均改进**: {np.mean(improvements):.2f}%\n")
                f.write(f"- **最大改进**: {np.max(improvements):.2f}%\n")
                f.write(f"- **最小改进**: {np.min(improvements):.2f}%\n")
            
            f.write("\n### 详细结果\n\n")
            for result in exp1_results:
                f.write(f"#### {result.design_name}\n\n")
                f.write(f"- **状态**: {'✅ 成功' if result.success else '❌ 失败'}\n")
                f.write(f"- **基线HPWL**: {result.baseline_hpwl:.2e}\n")
                f.write(f"- **优化HPWL**: {result.optimized_hpwl:.2e}\n")
                f.write(f"- **改进**: {result.improvement:.2f}%\n")
                f.write(f"- **执行时间**: {result.execution_time:.2f}秒\n")
                if result.error:
                    f.write(f"- **错误**: {result.error}\n")
                f.write("\n")
            
            # 实验2报告
            f.write("## 实验2: 跨设计泛化验证\n\n")
            exp2_results = results["experiment_2_cross_design"]
            
            successful_exp2 = [r for r in exp2_results if r.success]
            failed_exp2 = [r for r in exp2_results if not r.success]
            
            f.write(f"- **总实验数**: {len(exp2_results)}\n")
            f.write(f"- **成功数**: {len(successful_exp2)}\n")
            f.write(f"- **失败数**: {len(failed_exp2)}\n")
            
            if successful_exp2:
                improvements = [r.improvement for r in successful_exp2]
                f.write(f"- **平均改进**: {np.mean(improvements):.2f}%\n")
                f.write(f"- **最大改进**: {np.max(improvements):.2f}%\n")
                f.write(f"- **最小改进**: {np.min(improvements):.2f}%\n")
            
            f.write("\n### 详细结果\n\n")
            for result in exp2_results:
                f.write(f"#### {result.source_design} → {result.target_design}\n\n")
                f.write(f"- **状态**: {'✅ 成功' if result.success else '❌ 失败'}\n")
                f.write(f"- **基线HPWL**: {result.baseline_hpwl:.2e}\n")
                f.write(f"- **优化HPWL**: {result.optimized_hpwl:.2e}\n")
                f.write(f"- **改进**: {result.improvement:.2f}%\n")
                f.write(f"- **执行时间**: {result.execution_time:.2f}秒\n")
                if result.error:
                    f.write(f"- **错误**: {result.error}\n")
                f.write("\n")
        
        logger.info(f"实验报告已生成: {report_file}")

def main():
    """主函数"""
    # 创建实验实例
    experiment = RLOptimizationExperiment()
    
    # 运行所有实验
    results = experiment.run_all_experiments()
    
    # 输出总结
    logger.info("=== 实验总结 ===")
    
    exp1_results = results["experiment_1_same_design"]
    exp2_results = results["experiment_2_cross_design"]
    
    successful_exp1 = [r for r in exp1_results if r.success]
    successful_exp2 = [r for r in exp2_results if r.success]
    
    logger.info(f"实验1 (同设计优化): {len(successful_exp1)}/{len(exp1_results)} 成功")
    if successful_exp1:
        improvements = [r.improvement for r in successful_exp1]
        logger.info(f"  平均改进: {np.mean(improvements):.2f}%")
    
    logger.info(f"实验2 (跨设计泛化): {len(successful_exp2)}/{len(exp2_results)} 成功")
    if successful_exp2:
        improvements = [r.improvement for r in successful_exp2]
        logger.info(f"  平均改进: {np.mean(improvements):.2f}%")
    
    logger.info("实验完成！")

if __name__ == "__main__":
    main() 