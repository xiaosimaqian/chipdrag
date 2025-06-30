#!/usr/bin/env python3
"""
离线强化学习训练器
从已有数据中学习布局参数优化策略
"""

import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from .rl_training_system import LayoutAction, LayoutState

logger = logging.getLogger(__name__)

class OfflineRLTrainer:
    """离线强化学习训练器"""
    
    def __init__(self, 
                 data_dir: str = "results/parallel_training",
                 model_save_dir: str = "models/offline_rl"):
        """
        Args:
            data_dir: 训练数据目录
            model_save_dir: 模型保存目录
        """
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据预处理
        self.scaler = StandardScaler()
        self.training_data = []
        self.validation_data = []
        
        # 模型组件
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        
    def load_training_data(self, data_files: List[str] = None) -> pd.DataFrame:
        """加载训练数据"""
        logger.info("加载离线训练数据...")
        
        all_data = []
        
        # 如果没有指定文件，自动查找
        if data_files is None:
            # 查找所有可能的数据文件
            data_files = []
            
            # 查找批量训练结果
            batch_results = list(self.data_dir.glob("*.json"))
            data_files.extend([str(f) for f in batch_results])
            
            # 查找其他可能的训练数据
            other_data = list(self.data_dir.rglob("*.json"))
            data_files.extend([str(f) for f in other_data])
        
        for data_file in data_files:
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # 处理不同格式的数据
                if isinstance(data, list):
                    all_data.extend(self._process_data_list(data))
                elif isinstance(data, dict):
                    processed_data = self._process_data_dict(data)
                    if processed_data:
                        all_data.append(processed_data)
                        
            except Exception as e:
                logger.warning(f"加载数据文件 {data_file} 失败: {e}")
                continue
        
        if not all_data:
            logger.error("未找到有效的训练数据")
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        logger.info(f"成功加载 {len(df)} 条训练数据")
        
        return df
    
    def _process_data_list(self, data_list: List[Dict]) -> List[Dict]:
        """处理数据列表"""
        processed_data = []
        
        for item in data_list:
            processed_item = self._extract_training_sample(item)
            if processed_item:
                processed_data.append(processed_item)
        
        return processed_data
    
    def _process_data_dict(self, data_dict: Dict) -> Optional[Dict]:
        """处理数据字典"""
        return self._extract_training_sample(data_dict)
    
    def _extract_training_sample(self, item: Dict) -> Optional[Dict]:
        """从数据项中提取训练样本"""
        try:
            # 尝试提取参数
            parameters = {}
            
            # 从不同字段中提取参数
            if 'parameters' in item:
                parameters = item['parameters']
            elif 'action' in item:
                parameters = item['action']
            elif 'density_target' in item:
                parameters = {
                    'density_target': item.get('density_target', 0.7),
                    'wirelength_weight': item.get('wirelength_weight', 1.0),
                    'density_weight': item.get('density_weight', 1.0)
                }
            
            # 尝试提取性能指标
            performance = {}
            
            if 'hpwl' in item:
                performance['hpwl'] = item['hpwl']
            elif 'results' in item and 'hpwl' in item['results']:
                performance['hpwl'] = item['results']['hpwl']
            
            if 'overflow' in item:
                performance['overflow'] = item['overflow']
            elif 'results' in item and 'overflow' in item['results']:
                performance['overflow'] = item['results']['overflow']
            
            # 尝试提取设计特征
            design_features = {}
            
            if 'design_stats' in item:
                design_features = item['design_stats']
            elif 'num_instances' in item:
                design_features = {
                    'num_instances': item.get('num_instances', 0),
                    'num_nets': item.get('num_nets', 0),
                    'num_pins': item.get('num_pins', 0)
                }
            
            # 检查数据完整性
            if not parameters or not performance or 'hpwl' not in performance:
                return None
            
            # 构建训练样本
            sample = {
                'parameters': parameters,
                'performance': performance,
                'design_features': design_features,
                'design_name': item.get('design', 'unknown'),
                'success': item.get('success', True)
            }
            
            return sample
            
        except Exception as e:
            logger.debug(f"提取训练样本失败: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预处理数据"""
        logger.info("预处理训练数据...")
        
        # 提取特征和标签
        X_features = []  # 设计特征
        X_actions = []   # 动作参数
        y_rewards = []   # 奖励（基于HPWL）
        
        for _, row in df.iterrows():
            try:
                # 设计特征
                design_features = row['design_features']
                features = [
                    design_features.get('num_instances', 0),
                    design_features.get('num_nets', 0),
                    design_features.get('num_pins', 0)
                ]
                
                # 动作参数
                parameters = row['parameters']
                actions = [
                    parameters.get('density_target', 0.7),
                    parameters.get('wirelength_weight', 1.0),
                    parameters.get('density_weight', 1.0)
                ]
                
                # 计算奖励
                hpwl = row['performance']['hpwl']
                overflow = row['performance'].get('overflow', 0.1)
                success = row['success']
                
                # 奖励函数：HPWL越小越好，溢出率越低越好
                if success and hpwl > 0:
                    # 归一化HPWL（假设范围在1e6到1e10之间）
                    normalized_hpwl = np.log10(max(hpwl, 1e6)) - 6
                    reward = -normalized_hpwl - 10 * overflow
                else:
                    reward = -100  # 失败惩罚
                
                X_features.append(features)
                X_actions.append(actions)
                y_rewards.append(reward)
                
            except Exception as e:
                logger.debug(f"处理数据行失败: {e}")
                continue
        
        if not X_features:
            logger.error("没有有效的训练样本")
            return np.array([]), np.array([]), np.array([])
        
        # 转换为numpy数组
        X_features = np.array(X_features, dtype=np.float32)
        X_actions = np.array(X_actions, dtype=np.float32)
        y_rewards = np.array(y_rewards, dtype=np.float32)
        
        # 标准化特征
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        logger.info(f"预处理完成: {len(X_features_scaled)} 个样本")
        
        return X_features_scaled, X_actions, y_rewards
    
    def create_model(self, input_size: int, hidden_size: int = 128) -> nn.Module:
        """创建神经网络模型"""
        class ParameterPredictor(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 3)  # 输出3个参数
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.sigmoid(self.fc3(x))  # 输出在[0,1]范围内
                return x
        
        model = ParameterPredictor(input_size, hidden_size).to(self.device)
        return model
    
    def train_model(self, 
                   X_features: np.ndarray, 
                   X_actions: np.ndarray, 
                   y_rewards: np.ndarray,
                   epochs: int = 100,
                   batch_size: int = 32,
                   learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """训练模型"""
        logger.info("开始离线RL模型训练...")
        
        # 分割训练集和验证集
        X_train_feat, X_val_feat, X_train_act, X_val_act, y_train, y_val = train_test_split(
            X_features, X_actions, y_rewards, test_size=0.2, random_state=42
        )
        
        # 创建模型
        input_size = X_features.shape[1] + X_actions.shape[1]  # 特征 + 动作
        self.model = self.create_model(input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(np.hstack([X_train_feat, X_train_act])),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(np.hstack([X_val_feat, X_val_act])),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_rewards in train_loader:
                batch_features = batch_features.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                # 前向传播
                predicted_rewards = self.model(batch_features).squeeze()
                loss = criterion(predicted_rewards, batch_rewards)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_rewards in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_rewards = batch_rewards.to(self.device)
                    
                    predicted_rewards = self.model(batch_features).squeeze()
                    loss = criterion(predicted_rewards, batch_rewards)
                    val_loss += loss.item()
            
            # 记录损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}")
        
        # 保存训练历史
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
        
        logger.info("离线RL模型训练完成")
        return training_history
    
    def predict_optimal_parameters(self, design_features: Dict[str, Any]) -> LayoutAction:
        """预测最优参数"""
        if self.model is None:
            logger.error("模型未训练，无法预测参数")
            return LayoutAction.random_action()
        
        # 准备设计特征
        features = [
            design_features.get('num_instances', 0),
            design_features.get('num_nets', 0),
            design_features.get('num_pins', 0)
        ]
        
        # 标准化特征
        features_scaled = self.scaler.transform([features])
        
        # 使用网格搜索找到最优参数
        best_reward = float('-inf')
        best_action = None
        
        # 参数搜索空间
        density_targets = [0.6, 0.7, 0.8, 0.9]
        wirelength_weights = [0.5, 1.0, 2.0, 3.0]
        density_weights = [0.5, 1.0, 2.0, 3.0]
        
        self.model.eval()
        with torch.no_grad():
            for density_target in density_targets:
                for wirelength_weight in wirelength_weights:
                    for density_weight in density_weights:
                        # 构建输入
                        actions = [density_target, wirelength_weight, density_weight]
                        input_data = np.hstack([features_scaled[0], actions])
                        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
                        
                        # 预测奖励
                        predicted_reward = self.model(input_tensor).item()
                        
                        if predicted_reward > best_reward:
                            best_reward = predicted_reward
                            best_action = LayoutAction(
                                density_target=density_target,
                                wirelength_weight=wirelength_weight,
                                density_weight=density_weight,
                                overflow_penalty=0.0001,
                                max_displacement=5.0
                            )
        
        if best_action is None:
            best_action = LayoutAction.random_action()
        
        logger.info(f"预测最优参数: {best_action.to_dict()}, 预测奖励: {best_reward:.4f}")
        return best_action
    
    def save_model(self, model_name: str = "offline_rl_model"):
        """保存模型"""
        if self.model is None:
            logger.error("没有训练好的模型可保存")
            return
        
        model_path = self.model_save_dir / f"{model_name}.pth"
        scaler_path = self.model_save_dir / f"{model_name}_scaler.pkl"
        
        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, model_path)
        
        # 保存标准化器
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"模型已保存: {model_path}")
    
    def load_model(self, model_name: str = "offline_rl_model"):
        """加载模型"""
        model_path = self.model_save_dir / f"{model_name}.pth"
        scaler_path = self.model_save_dir / f"{model_name}_scaler.pkl"
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            input_size = 6  # 3个特征 + 3个动作
            self.model = self.create_model(input_size)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载标准化器
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"模型已加载: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def generate_training_report(self, training_history: Dict, save_dir: Path):
        """生成训练报告"""
        # 绘制训练曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_history['train_losses'], label='Train Loss')
        plt.plot(training_history['val_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(training_history['train_losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(training_history['val_losses'])
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.bar(['Final Train Loss', 'Final Val Loss'], 
                [training_history['final_train_loss'], training_history['final_val_loss']])
        plt.title('Final Loss Comparison')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(save_dir / "offline_rl_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存训练历史
        with open(save_dir / "offline_rl_training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info(f"训练报告已保存到: {save_dir}")

def main():
    """主函数 - 离线RL训练示例"""
    # 配置参数
    data_dir = "results/parallel_training"
    model_save_dir = "models/offline_rl"
    
    # 创建离线RL训练器
    trainer = OfflineRLTrainer(data_dir, model_save_dir)
    
    # 加载训练数据
    df = trainer.load_training_data()
    
    if df.empty:
        logger.error("没有找到训练数据，请先运行批量训练")
        return
    
    # 预处理数据
    X_features, X_actions, y_rewards = trainer.preprocess_data(df)
    
    if len(X_features) == 0:
        logger.error("预处理后没有有效数据")
        return
    
    # 训练模型
    training_history = trainer.train_model(
        X_features, X_actions, y_rewards,
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    # 保存模型
    trainer.save_model()
    
    # 生成训练报告
    trainer.generate_training_report(training_history, Path(model_save_dir))
    
    # 测试参数预测
    test_design_features = {
        'num_instances': 50000,
        'num_nets': 60000,
        'num_pins': 1000
    }
    
    optimal_params = trainer.predict_optimal_parameters(test_design_features)
    logger.info(f"测试设计的最优参数: {optimal_params.to_dict()}")
    
    logger.info("离线RL训练完成！")

if __name__ == "__main__":
    main() 