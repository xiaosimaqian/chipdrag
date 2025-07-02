#!/usr/bin/env python3
"""
扩充训练案例库
将RL训练episode数据转换为案例库，增加LLM知识检索的案例数量
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingCaseExpander:
    """训练案例扩充器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data/designs/ispd_2015_contest_benchmark"
        self.knowledge_dir = self.base_dir / "data/knowledge_base/layout_experience"
        
        # 确保知识库目录存在
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_episode_data(self) -> List[Dict[str, Any]]:
        """收集所有设计的episode数据"""
        logger.info("开始收集episode数据...")
        
        all_episodes = []
        design_count = 0
        
        for design_dir in self.data_dir.iterdir():
            if not design_dir.is_dir():
                continue
                
            design_name = design_dir.name
            rl_training_dir = design_dir / "rl_training"
            
            if not rl_training_dir.exists():
                continue
                
            # 查找episode文件
            episode_files = list(rl_training_dir.glob("episode_*.json"))
            if not episode_files:
                continue
                
            logger.info(f"处理设计 {design_name}: 找到 {len(episode_files)} 个episode文件")
            design_count += 1
            
            for episode_file in episode_files:
                try:
                    with open(episode_file, 'r', encoding='utf-8') as f:
                        episode_steps = json.load(f)
                    
                    # episode_steps是一个数组，包含多个步骤
                    if isinstance(episode_steps, list):
                        for step_idx, step_data in enumerate(episode_steps):
                            # 添加设计信息和步骤信息
                            step_data['design_name'] = design_name
                            step_data['episode_file'] = episode_file.name
                            step_data['step_index'] = step_idx
                            step_data['episode_id'] = int(episode_file.stem.split('_')[1])
                            
                            all_episodes.append(step_data)
                    else:
                        # 如果是单个episode数据
                        episode_steps['design_name'] = design_name
                        episode_steps['episode_file'] = episode_file.name
                        all_episodes.append(episode_steps)
                    
                except Exception as e:
                    logger.warning(f"读取episode文件失败 {episode_file}: {e}")
        
        logger.info(f"总共收集到 {len(all_episodes)} 个episode，来自 {design_count} 个设计")
        return all_episodes
    
    def convert_episodes_to_cases(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将episode数据转换为案例格式"""
        logger.info("开始转换episode为案例...")
        
        cases = []
        
        for i, episode in enumerate(episodes):
            try:
                design_name = episode.get('design_name', f'design_{i}')
                state = episode.get('state', [])
                action = episode.get('action', [])
                reward = episode.get('reward', 0.0)
                done = episode.get('done', False)
                # 直接存list
                layout = {
                    'design_name': design_name,
                    'state_vector': state,
                    'metadata': {
                        'episode_id': episode.get('episode_id', i),
                        'step': episode.get('step_index', 0),
                        'timestamp': episode.get('timestamp', datetime.now().isoformat())
                    }
                }
                optimization_result = {
                    'action_vector': action,
                    'reward': reward,
                    'done': done,
                    'success': reward > 0
                }
                case = {
                    'id': len(cases),
                    'layout': layout,
                    'optimization_result': optimization_result,
                    'metadata': {
                        'source': 'rl_training_episode',
                        'design_name': design_name,
                        'episode_file': episode.get('episode_file', ''),
                        'conversion_timestamp': datetime.now().isoformat()
                    },
                    'timestamp': episode.get('timestamp', datetime.now().isoformat())
                }
                cases.append(case)
            except Exception as e:
                logger.warning(f"转换episode {i} 失败: {e}")
                continue
        
        logger.info(f"成功转换 {len(cases)} 个案例")
        return cases
    
    def load_existing_cases(self) -> List[Dict[str, Any]]:
        """加载现有案例"""
        cases_file = self.knowledge_dir / "cases.pkl"
        
        if cases_file.exists():
            try:
                with open(cases_file, 'rb') as f:
                    existing_cases = pickle.load(f)
                logger.info(f"加载现有案例 {len(existing_cases)} 个")
                return existing_cases
            except Exception as e:
                logger.warning(f"加载现有案例失败: {e}")
        
        return []
    
    def merge_and_save_cases(self, new_cases: List[Dict[str, Any]], existing_cases: List[Dict[str, Any]]):
        """合并并保存案例"""
        logger.info("合并案例...")
        
        # 合并案例
        all_cases = existing_cases + new_cases
        
        # 去重（基于设计名称和episode信息）
        unique_cases = []
        seen_keys = set()
        
        for case in all_cases:
            design_name = case.get('metadata', {}).get('design_name', '')
            episode_file = case.get('metadata', {}).get('episode_file', '')
            key = f"{design_name}_{episode_file}_{case.get('id', 0)}"
            
            if key not in seen_keys:
                seen_keys.add(key)
                unique_cases.append(case)
        
        logger.info(f"去重后案例数量: {len(unique_cases)} (原有: {len(existing_cases)}, 新增: {len(new_cases)})")
        
        # 保存案例
        cases_file = self.knowledge_dir / "cases.pkl"
        with open(cases_file, 'wb') as f:
            pickle.dump(unique_cases, f)
        
        logger.info(f"案例已保存到: {cases_file}")
        
        # 生成案例统计
        self._generate_case_statistics(unique_cases)
        
        return unique_cases
    
    def _generate_case_statistics(self, cases: List[Dict[str, Any]]):
        """生成案例统计信息"""
        logger.info("生成案例统计...")
        
        stats = {
            'total_cases': len(cases),
            'designs': {},
            'sources': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for case in cases:
            # 按设计统计
            design_name = case.get('metadata', {}).get('design_name', 'unknown')
            if design_name not in stats['designs']:
                stats['designs'][design_name] = 0
            stats['designs'][design_name] += 1
            
            # 按来源统计
            source = case.get('metadata', {}).get('source', 'unknown')
            if source not in stats['sources']:
                stats['sources'][source] = 0
            stats['sources'][source] += 1
        
        # 保存统计
        stats_file = self.knowledge_dir / "case_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"案例统计已保存: {stats_file}")
        logger.info(f"案例总数: {stats['total_cases']}")
        logger.info(f"涉及设计: {list(stats['designs'].keys())}")
        logger.info(f"数据来源: {stats['sources']}")
    
    def expand_cases(self):
        """扩充案例库的主流程"""
        logger.info("=== 开始扩充训练案例库 ===")
        
        # 1. 收集episode数据
        episodes = self.collect_episode_data()
        
        if not episodes:
            logger.warning("未找到任何episode数据")
            return
        
        # 2. 转换为案例
        new_cases = self.convert_episodes_to_cases(episodes)
        
        if not new_cases:
            logger.warning("未能转换任何案例")
            return
        
        # 3. 加载现有案例
        existing_cases = self.load_existing_cases()
        
        # 4. 合并并保存
        all_cases = self.merge_and_save_cases(new_cases, existing_cases)
        
        logger.info("=== 案例库扩充完成 ===")
        logger.info(f"最终案例总数: {len(all_cases)}")

def main():
    """主函数"""
    expander = TrainingCaseExpander()
    expander.expand_cases()

if __name__ == "__main__":
    main() 