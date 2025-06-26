#!/usr/bin/env python3
"""
RL与OpenROAD联动实验系统
实现强化学习智能体与OpenROAD的集成，支持多参数实验和批量训练
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt

# 导入统一OpenROAD接口
from simple_openroad_test import run_openroad_with_docker

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LayoutState:
    """布局状态空间"""
    def __init__(self, design_path: Path):
        self.design_features = self._extract_design_features(design_path)
        self.retrieval_features = {
            "current_k": 5,
            "retrieval_history": [],
            "query_complexity": 0.5,
            "knowledge_coverage": 0.0
        }
        self.quality_features = {
            "current_wirelength": 0.0,
            "current_congestion": 0.0,
            "current_timing": 0.0,
            "current_power": 0.0,
            "improvement_rate": 0.0
        }
    
    def _extract_design_features(self, design_path: Path) -> Dict:
        """提取设计特征"""
        # 读取verilog文件获取单元和网络数量
        v_file = design_path / "design.v"
        cell_count = 0
        net_count = 0
        
        if v_file.exists():
            with open(v_file, 'r') as f:
                content = f.read()
                # 简单统计模块实例数量
                cell_count = content.count('module') + content.count('cell')
                net_count = content.count('wire') + content.count('assign')
        
        return {
            "cell_count": cell_count,
            "net_count": net_count,
            "design_area": 100.0,  # 默认值，可从def文件解析
            "constraint_count": 5,  # 默认值
            "design_type": design_path.name.split('_')[0] if '_' in design_path.name else "unknown",
            "complexity_score": min(1.0, (cell_count + net_count) / 10000.0)
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "design_features": self.design_features,
            "retrieval_features": self.retrieval_features,
            "quality_features": self.quality_features
        }
    
    def update_quality(self, quality_metrics: Dict):
        """更新质量特征"""
        self.quality_features.update(quality_metrics)

class LayoutAction:
    """布局动作空间"""
    def __init__(self):
        self.retrieval_actions = {
            "k_value": random.choice(range(3, 16)),
            "retrieval_strategy": random.choice([
                "semantic", "constraint_based", "experience_based", "hybrid"
            ])
        }
        self.placement_actions = {
            "density_target": random.choice([0.7, 0.8, 0.9, 0.95]),
            "placement_algorithm": random.choice([
                "global_placement", "detailed_placement", "incremental_placement"
            ]),
            "optimization_focus": random.choice([
                "wirelength", "congestion", "timing", "power", "balanced"
            ])
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "retrieval_actions": self.retrieval_actions,
            "placement_actions": self.placement_actions
        }

class LayoutRLAgent:
    """布局强化学习智能体"""
    def __init__(self):
        self.q_table = {}  # Q值表
        self.epsilon = 0.9  # 探索率
        self.alpha = 0.01   # 学习率
        self.gamma = 0.95   # 折扣因子
        self.experience_buffer = []  # 经验回放缓冲区
        
    def get_state_key(self, state: LayoutState) -> str:
        """获取状态键值"""
        # 简化的状态表示
        features = state.design_features
        return f"{features['design_type']}_{features['cell_count']}_{features['net_count']}"
    
    def select_action(self, state: LayoutState) -> LayoutAction:
        """选择动作（ε-贪婪策略）"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return LayoutAction()
        else:
            # 利用：选择Q值最大的动作
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # 简化的动作选择（实际应该遍历所有可能动作）
            return LayoutAction()
    
    def update_q_value(self, state: LayoutState, action: LayoutAction, reward: float, next_state: LayoutState):
        """更新Q值"""
        state_key = self.get_state_key(state)
        action_key = str(action.to_dict())
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Q-learning更新公式
        next_state_key = self.get_state_key(next_state)
        max_next_q = 0.0
        if next_state_key in self.q_table:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        current_q = self.q_table[state_key][action_key]
        self.q_table[state_key][action_key] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """衰减探索率"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)

class TCLScriptGenerator:
    """TCL脚本生成器"""
    def __init__(self):
        pass
    
    def generate_placement_script(self, state: LayoutState, action: LayoutAction) -> str:
        """生成布局TCL脚本"""
        script = f"""
# 动态布局脚本 - 基于RL状态和动作生成
puts "开始动态布局优化..."

# 读取设计文件
read_lef tech.lef
read_lef cells.lef
read_def floorplan.def
read_verilog design.v

# 链接设计
link_design des_perf

# 设置布局参数 (来自RL动作)
set density_target {action.placement_actions['density_target']}
set optimization_focus "{action.placement_actions['optimization_focus']}"

# 执行布局
puts "执行{action.placement_actions['placement_algorithm']}..."
{action.placement_actions['placement_algorithm']} -density $density_target

# 根据优化重点进行特定优化
if {{$optimization_focus == "wirelength"}} {{
    optimize_wirelength
}} elseif {{$optimization_focus == "congestion"}} {{
    optimize_congestion
}} elseif {{$optimization_focus == "timing"}} {{
    optimize_timing
}} elseif {{$optimization_focus == "power"}} {{
    optimize_power
}}

# 输出质量指标
report_wirelength
report_congestion
report_timing
report_power

# 保存结果
write_def final_placement.def
puts "布局完成"
"""
        return script

class OpenROADResultParser:
    """OpenROAD结果解析器"""
    def __init__(self):
        pass
    
    def parse_placement_output(self, output: str) -> Dict:
        """解析布局输出 - 只解析真实数据，无默认值"""
        metrics = {}
        
        # 解析线长
        if "Total wirelength:" in output:
            try:
                wirelength_line = [line for line in output.split('\n') if "Total wirelength:" in line][0]
                wirelength = float(wirelength_line.split(":")[-1].strip())
                metrics['wirelength'] = wirelength
            except Exception as e:
                logger.error(f"解析线长失败: {e}")
                raise ValueError(f"无法解析OpenROAD输出的线长信息: {output[:200]}...")
        else:
            logger.error("OpenROAD输出中未找到线长信息")
            raise ValueError("OpenROAD输出中未找到线长信息")
        
        # 解析拥塞
        if "Congestion:" in output:
            try:
                congestion_line = [line for line in output.split('\n') if "Congestion:" in line][0]
                congestion = float(congestion_line.split(":")[-1].strip())
                metrics['congestion'] = congestion
            except Exception as e:
                logger.error(f"解析拥塞失败: {e}")
                raise ValueError(f"无法解析OpenROAD输出的拥塞信息: {output[:200]}...")
        else:
            logger.error("OpenROAD输出中未找到拥塞信息")
            raise ValueError("OpenROAD输出中未找到拥塞信息")
        
        # 解析时序
        if "Worst slack:" in output:
            try:
                timing_line = [line for line in output.split('\n') if "Worst slack:" in line][0]
                timing_slack = float(timing_line.split(":")[-1].strip())
                metrics['timing_slack'] = timing_slack
            except Exception as e:
                logger.error(f"解析时序失败: {e}")
                raise ValueError(f"无法解析OpenROAD输出的时序信息: {output[:200]}...")
        else:
            logger.error("OpenROAD输出中未找到时序信息")
            raise ValueError("OpenROAD输出中未找到时序信息")
        
        # 解析功耗
        if "Total power:" in output:
            try:
                power_line = [line for line in output.split('\n') if "Total power:" in line][0]
                power = float(power_line.split(":")[-1].strip())
                metrics['power'] = power
            except Exception as e:
                logger.error(f"解析功耗失败: {e}")
                raise ValueError(f"无法解析OpenROAD输出的功耗信息: {output[:200]}...")
        else:
            logger.error("OpenROAD输出中未找到功耗信息")
            raise ValueError("OpenROAD输出中未找到功耗信息")
        
        return metrics

class LayoutRLExperiment:
    """布局RL实验主控类"""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent = LayoutRLAgent()
        self.script_generator = TCLScriptGenerator()
        self.result_parser = OpenROADResultParser()
        
        self.training_history = []
        self.episode_results = []
    
    def calculate_reward(self, quality_metrics: Dict) -> float:
        """计算奖励函数 - 基于真实指标"""
        # 确保所有指标都存在
        required_metrics = ['wirelength', 'congestion', 'timing_slack', 'power']
        for metric in required_metrics:
            if metric not in quality_metrics:
                raise ValueError(f"缺少必需的指标: {metric}")
        
        # 多目标奖励函数 - 基于真实指标
        wirelength_reward = max(0, 1.0 - quality_metrics['wirelength'] / 10000)
        congestion_reward = max(0, 1.0 - quality_metrics['congestion'])
        timing_reward = max(0, quality_metrics['timing_slack'])
        power_reward = max(0, 1.0 - quality_metrics['power'])
        
        # 加权组合
        total_reward = (
            0.3 * wirelength_reward +
            0.25 * congestion_reward +
            0.25 * timing_reward +
            0.2 * power_reward
        )
        
        return total_reward
    
    def execute_action(self, state: LayoutState, action: LayoutAction, design_path: Path) -> Tuple[LayoutState, float]:
        """执行动作 - 调用OpenROAD"""
        # 1. 生成TCL脚本
        tcl_script = self.script_generator.generate_placement_script(state, action)
        
        # 2. 保存TCL脚本
        tcl_path = design_path / "rl_placement.tcl"
        with open(tcl_path, 'w') as f:
            f.write(tcl_script)
            f.flush()
            os.fsync(f.fileno())
        
        # 3. 调用OpenROAD
        result = run_openroad_with_docker(
            work_dir=design_path,
            cmd=tcl_path.name,
            is_tcl=True
        )
        
        # 4. 检查OpenROAD执行结果
        if result.returncode != 0:
            logger.error(f"OpenROAD执行失败，返回码: {result.returncode}")
            logger.error(f"错误输出: {result.stderr}")
            raise RuntimeError(f"OpenROAD执行失败: {result.stderr}")
        
        # 5. 解析结果 - 如果解析失败会抛出异常
        quality_metrics = self.result_parser.parse_placement_output(result.stdout)
        
        # 6. 计算奖励
        reward = self.calculate_reward(quality_metrics)
        
        # 7. 构建下一状态
        next_state = LayoutState(design_path)
        next_state.update_quality(quality_metrics)
        
        return next_state, reward
    
    def train_episode(self, design_path: Path, max_steps: int = 10) -> Dict:
        """训练单个episode"""
        state = LayoutState(design_path)
        episode_reward = 0.0
        episode_steps = []
        
        for step in range(max_steps):
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward = self.execute_action(state, action, design_path)
            
            # 更新Q值
            self.agent.update_q_value(state, action, reward, next_state)
            
            # 记录步骤
            step_info = {
                "step": step,
                "state": state.to_dict(),
                "action": action.to_dict(),
                "reward": reward,
                "quality_metrics": next_state.quality_features
            }
            episode_steps.append(step_info)
            
            episode_reward += reward
            state = next_state
            
            # 检查终止条件
            if reward > 0.8:  # 达到高质量阈值
                break
        
        # 衰减探索率
        self.agent.decay_epsilon()
        
        episode_result = {
            "design": design_path.name,
            "episode_reward": episode_reward,
            "steps": episode_steps,
            "final_quality": state.quality_features
        }
        
        self.episode_results.append(episode_result)
        return episode_result
    
    def run_experiment(self, benchmark_designs: List[Path], num_episodes: int = 100) -> Dict:
        """运行完整实验"""
        logger.info(f"开始RL实验，{len(benchmark_designs)}个设计，{num_episodes}个episode")
        
        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            
            # 随机选择一个设计
            design_path = random.choice(benchmark_designs)
            
            # 训练episode
            episode_result = self.train_episode(design_path)
            
            # 记录训练历史
            self.training_history.append({
                "episode": episode,
                "reward": episode_result["episode_reward"],
                "epsilon": self.agent.epsilon
            })
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([h["reward"] for h in self.training_history[-10:]])
                logger.info(f"Episode {episode + 1}, 平均奖励: {avg_reward:.4f}, ε: {self.agent.epsilon:.4f}")
        
        # 保存结果
        self.save_results()
        
        return {
            "training_history": self.training_history,
            "episode_results": self.episode_results,
            "final_q_table": self.agent.q_table
        }
    
    def save_results(self):
        """保存实验结果"""
        # 保存训练历史
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存episode结果
        with open(self.output_dir / "episode_results.json", 'w') as f:
            json.dump(self.episode_results, f, indent=2)
        
        # 保存Q表
        with open(self.output_dir / "q_table.json", 'w') as f:
            json.dump(self.agent.q_table, f, indent=2)
        
        # 生成训练曲线
        self.plot_training_curves()
        
        # 生成CSV报告
        self.generate_csv_report()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        episodes = [h["episode"] for h in self.training_history]
        rewards = [h["reward"] for h in self.training_history]
        epsilons = [h["epsilon"] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 奖励曲线
        ax1.plot(episodes, rewards, 'b-', alpha=0.6)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Reward Curve')
        ax1.grid(True)
        
        # 探索率曲线
        ax2.plot(episodes, epsilons, 'r-')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        ax2.set_title('Exploration Rate Decay')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_csv_report(self):
        """生成CSV报告"""
        # 训练历史CSV
        df_history = pd.DataFrame(self.training_history)
        df_history.to_csv(self.output_dir / "training_history.csv", index=False)
        
        # Episode结果CSV
        episode_data = []
        for episode in self.episode_results:
            episode_data.append({
                "design": episode["design"],
                "episode_reward": episode["episode_reward"],
                "final_wirelength": episode["final_quality"]["current_wirelength"],
                "final_congestion": episode["final_quality"]["current_congestion"],
                "final_timing": episode["final_quality"]["current_timing"],
                "final_power": episode["final_quality"]["current_power"]
            })
        
        df_episodes = pd.DataFrame(episode_data)
        df_episodes.to_csv(self.output_dir / "episode_results.csv", index=False)

def main():
    """主函数"""
    # 设置输出目录
    output_dir = Path("results/rl_experiment")
    
    # 获取ISPD基准设计
    benchmark_dir = Path("../data/designs/ispd_2015_contest_benchmark")
    benchmark_designs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
    
    # 创建实验系统
    experiment = LayoutRLExperiment(output_dir)
    
    # 运行实验
    results = experiment.run_experiment(benchmark_designs, num_episodes=50)
    
    logger.info("RL实验完成！结果保存在: " + str(output_dir))

if __name__ == "__main__":
    main() 