#!/usr/bin/env python3
"""
综合案例收集器
整合所有能找到的案例数据，包括：
1. ispd_cases.json (16个案例)
2. layout_experience/cases.pkl (151个案例)
3. 所有episode数据 (30个episode文件)
4. 训练结果数据
5. 其他可能的案例来源
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveCaseCollector:
    """综合案例收集器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.knowledge_dir = self.base_dir / "data/knowledge_base"
        self.layout_experience_dir = self.knowledge_dir / "layout_experience"
        self.designs_dir = self.base_dir / "data/designs/ispd_2015_contest_benchmark"
        
        # 确保目录存在
        self.layout_experience_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于去重的集合
        self.seen_case_keys: Set[str] = set()
        self.all_cases: List[Dict[str, Any]] = []
        
    def collect_ispd_cases(self) -> int:
        """收集ispd_cases.json中的案例"""
        logger.info("收集ispd_cases.json中的案例...")
        
        ispd_cases_file = self.knowledge_dir / "ispd_cases.json"
        if not ispd_cases_file.exists():
            logger.warning("ispd_cases.json不存在")
            return 0
        
        try:
            with open(ispd_cases_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)
            
            added_count = 0
            for case in cases:
                case_key = f"ispd_{case.get('design_name', 'unknown')}_{case.get('case_id', 'unknown')}"
                if case_key not in self.seen_case_keys:
                    self.seen_case_keys.add(case_key)
                    case['source'] = 'ispd_cases.json'
                    case['collection_timestamp'] = datetime.now().isoformat()
                    self.all_cases.append(case)
                    added_count += 1
            
            logger.info(f"从ispd_cases.json添加了 {added_count} 个案例")
            return added_count
            
        except Exception as e:
            logger.error(f"读取ispd_cases.json失败: {e}")
            return 0
    
    def collect_layout_experience_cases(self) -> int:
        """收集layout_experience/cases.pkl中的案例"""
        logger.info("收集layout_experience/cases.pkl中的案例...")
        
        cases_file = self.layout_experience_dir / "cases.pkl"
        if not cases_file.exists():
            logger.warning("layout_experience/cases.pkl不存在")
            return 0
        
        try:
            with open(cases_file, 'rb') as f:
                cases = pickle.load(f)
            
            added_count = 0
            for case in cases:
                design_name = case.get('metadata', {}).get('design_name', 'unknown')
                episode_file = case.get('metadata', {}).get('episode_file', 'unknown')
                case_id = case.get('id', 'unknown')
                case_key = f"layout_exp_{design_name}_{episode_file}_{case_id}"
                
                if case_key not in self.seen_case_keys:
                    self.seen_case_keys.add(case_key)
                    case['source'] = 'layout_experience_cases.pkl'
                    case['collection_timestamp'] = datetime.now().isoformat()
                    self.all_cases.append(case)
                    added_count += 1
            
            logger.info(f"从layout_experience/cases.pkl添加了 {added_count} 个案例")
            return added_count
            
        except Exception as e:
            logger.error(f"读取layout_experience/cases.pkl失败: {e}")
            return 0
    
    def collect_episode_data(self) -> int:
        """收集所有episode数据"""
        logger.info("收集所有episode数据...")
        
        added_count = 0
        
        for design_dir in self.designs_dir.iterdir():
            if not design_dir.is_dir():
                continue
            
            design_name = design_dir.name
            rl_training_dir = design_dir / "rl_training"
            
            if not rl_training_dir.exists():
                continue
            
            episode_files = list(rl_training_dir.glob("episode_*.json"))
            if not episode_files:
                continue
            
            logger.info(f"处理设计 {design_name}: 找到 {len(episode_files)} 个episode文件")
            
            for episode_file in episode_files:
                try:
                    with open(episode_file, 'r', encoding='utf-8') as f:
                        episode_steps = json.load(f)
                    
                    # episode_steps是一个数组，包含多个步骤
                    if isinstance(episode_steps, list):
                        for step_idx, step_data in enumerate(episode_steps):
                            case_key = f"episode_{design_name}_{episode_file.stem}_{step_idx}"
                            
                            if case_key not in self.seen_case_keys:
                                self.seen_case_keys.add(case_key)
                                
                                # 转换为案例格式
                                case = {
                                    'id': len(self.all_cases),
                                    'design_name': design_name,
                                    'episode_file': episode_file.name,
                                    'step_index': step_idx,
                                    'state': step_data.get('state', []),
                                    'action': step_data.get('action', []),
                                    'reward': step_data.get('reward', 0.0),
                                    'done': step_data.get('done', False),
                                    'source': 'episode_data',
                                    'collection_timestamp': datetime.now().isoformat(),
                                    'metadata': {
                                        'design_name': design_name,
                                        'episode_file': episode_file.name,
                                        'step_index': step_idx,
                                        'episode_id': int(episode_file.stem.split('_')[1])
                                    }
                                }
                                self.all_cases.append(case)
                                added_count += 1
                    
                except Exception as e:
                    logger.warning(f"读取episode文件失败 {episode_file}: {e}")
        
        logger.info(f"从episode数据添加了 {added_count} 个案例")
        return added_count
    
    def collect_training_results(self) -> int:
        """收集训练结果数据"""
        logger.info("收集训练结果数据...")
        
        added_count = 0
        
        # 查找所有训练结果文件
        results_dirs = [
            self.base_dir / "results" / "ispd_training",
            self.base_dir / "results" / "ispd_training_fixed_v13"
        ]
        
        for results_dir in results_dirs:
            if not results_dir.exists():
                continue
            
            result_files = list(results_dir.glob("*_result.json"))
            logger.info(f"在 {results_dir} 中找到 {len(result_files)} 个结果文件")
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    design_name = result_file.stem.replace('_result', '')
                    case_key = f"training_result_{design_name}_{results_dir.name}"
                    
                    if case_key not in self.seen_case_keys:
                        self.seen_case_keys.add(case_key)
                        
                        case = {
                            'id': len(self.all_cases),
                            'design_name': design_name,
                            'source': f'training_result_{results_dir.name}',
                            'collection_timestamp': datetime.now().isoformat(),
                            'training_result': result_data,
                            'metadata': {
                                'design_name': design_name,
                                'result_file': result_file.name,
                                'results_dir': results_dir.name
                            }
                        }
                        self.all_cases.append(case)
                        added_count += 1
                
                except Exception as e:
                    logger.warning(f"读取训练结果文件失败 {result_file}: {e}")
        
        logger.info(f"从训练结果添加了 {added_count} 个案例")
        return added_count
    
    def collect_rl_agent_training_reports(self) -> int:
        """收集RL智能体训练报告"""
        logger.info("收集RL智能体训练报告...")
        
        added_count = 0
        rl_agent_dir = self.base_dir / "models" / "rl_agent"
        
        if not rl_agent_dir.exists():
            logger.warning("RL智能体目录不存在")
            return 0
        
        training_reports = list(rl_agent_dir.glob("training_report*.json"))
        logger.info(f"找到 {len(training_reports)} 个训练报告")
        
        for report_file in training_reports:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                design_name = report_file.stem.replace('training_report_', '').replace('training_report', 'general')
                case_key = f"rl_training_report_{design_name}"
                
                if case_key not in self.seen_case_keys:
                    self.seen_case_keys.add(case_key)
                    
                    case = {
                        'id': len(self.all_cases),
                        'design_name': design_name,
                        'source': 'rl_training_report',
                        'collection_timestamp': datetime.now().isoformat(),
                        'training_report': report_data,
                        'metadata': {
                            'design_name': design_name,
                            'report_file': report_file.name
                        }
                    }
                    self.all_cases.append(case)
                    added_count += 1
            
            except Exception as e:
                logger.warning(f"读取训练报告失败 {report_file}: {e}")
        
        logger.info(f"从RL训练报告添加了 {added_count} 个案例")
        return added_count
    
    def save_comprehensive_cases(self):
        """保存综合案例数据"""
        logger.info("保存综合案例数据...")
        
        # 保存为pickle格式
        cases_file = self.layout_experience_dir / "comprehensive_cases.pkl"
        with open(cases_file, 'wb') as f:
            pickle.dump(self.all_cases, f)
        
        logger.info(f"综合案例已保存到: {cases_file}")
        
        # 生成统计信息
        self._generate_comprehensive_statistics()
        
        # 同时更新原有的cases.pkl
        original_cases_file = self.layout_experience_dir / "cases.pkl"
        with open(original_cases_file, 'wb') as f:
            pickle.dump(self.all_cases, f)
        
        logger.info(f"原有cases.pkl已更新: {original_cases_file}")
    
    def _generate_comprehensive_statistics(self):
        """生成综合统计信息"""
        logger.info("生成综合统计信息...")
        
        stats = {
            'total_cases': len(self.all_cases),
            'sources': {},
            'designs': {},
            'collection_timestamp': datetime.now().isoformat()
        }
        
        for case in self.all_cases:
            # 按来源统计
            source = case.get('source', 'unknown')
            if source not in stats['sources']:
                stats['sources'][source] = 0
            stats['sources'][source] += 1
            
            # 按设计统计
            design_name = case.get('design_name', 'unknown')
            if design_name not in stats['designs']:
                stats['designs'][design_name] = 0
            stats['designs'][design_name] += 1
        
        # 保存统计
        stats_file = self.layout_experience_dir / "comprehensive_case_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"综合统计已保存: {stats_file}")
        logger.info(f"案例总数: {stats['total_cases']}")
        logger.info(f"数据来源: {stats['sources']}")
        logger.info(f"涉及设计数量: {len(stats['designs'])}")
    
    def collect_all_cases(self):
        """收集所有案例的主流程"""
        logger.info("=== 开始综合案例收集 ===")
        
        total_added = 0
        
        # 1. 收集ispd_cases.json
        total_added += self.collect_ispd_cases()
        
        # 2. 收集layout_experience/cases.pkl
        total_added += self.collect_layout_experience_cases()
        
        # 3. 收集episode数据
        total_added += self.collect_episode_data()
        
        # 4. 收集训练结果
        total_added += self.collect_training_results()
        
        # 5. 收集RL训练报告
        total_added += self.collect_rl_agent_training_reports()
        
        logger.info(f"=== 综合案例收集完成 ===")
        logger.info(f"总共收集到 {len(self.all_cases)} 个唯一案例")
        logger.info(f"新增案例数: {total_added}")
        
        # 保存综合案例
        self.save_comprehensive_cases()

def main():
    """主函数"""
    collector = ComprehensiveCaseCollector()
    collector.collect_all_cases()

if __name__ == "__main__":
    main() 