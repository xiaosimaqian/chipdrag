import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class MultimodalFusion:
    """多模态融合类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多模态融合器
        
        Args:
            config: 配置信息，包含各模态的权重
        """
        self.text_weight = config.get('text_weight', 0.4)
        self.image_weight = config.get('image_weight', 0.3)
        self.graph_weight = config.get('graph_weight', 0.3)
        
        # 添加weights属性
        self.weights = {
            'text': self.text_weight,
            'image': self.image_weight,
            'graph': self.graph_weight
        }
        
        # 验证权重和为1
        total_weight = self.text_weight + self.image_weight + self.graph_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"权重和不为1: {total_weight}，将进行归一化")
            self.text_weight /= total_weight
            self.image_weight /= total_weight
            self.graph_weight /= total_weight
            
            # 更新weights属性
            self.weights = {
                'text': self.text_weight,
                'image': self.image_weight,
                'graph': self.graph_weight
            }
            
        logger.info("多模态融合器初始化完成")
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征向量
        
        Args:
            features: 特征向量
            
        Returns:
            归一化后的特征向量
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return normalize(features, norm='l2', axis=1).squeeze()
        
    def _validate_features(self, features: np.ndarray, modality: str) -> np.ndarray:
        """验证特征向量的有效性
        
        Args:
            features: 特征向量
            modality: 模态名称
            
        Returns:
            验证和修复后的特征向量
            
        Raises:
            ValueError: 特征向量无效
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"{modality}特征必须是numpy数组")
        if features.size == 0:
            raise ValueError(f"{modality}特征不能为空")
        
        # 检查NaN值并替换为0
        if np.isnan(features).any():
            logger.warning(f"{modality}特征包含NaN值，将替换为0")
            features = np.nan_to_num(features, nan=0.0)
            
        # 检查无穷值并替换为0
        if np.isinf(features).any():
            logger.warning(f"{modality}特征包含无穷值，将替换为0")
            features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
            
        return features
        
    def _transform_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """转换特征维度
        
        Args:
            features: 特征向量
            target_dim: 目标维度
            
        Returns:
            转换后的特征向量
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # 如果特征维度大于目标维度，使用PCA降维
        if features.shape[1] > target_dim:
            # 确保样本数量足够
            n_samples = features.shape[0]
            n_components = min(target_dim, n_samples, features.shape[1])
            if n_components > 0:
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(features)
                # 如果降维后的维度小于目标维度，进行零填充
                if n_components < target_dim:
                    padded = np.zeros((transformed.shape[0], target_dim))
                    padded[:, :n_components] = transformed
                    return padded.squeeze()
                return transformed.squeeze()
            else:
                # 如果无法进行PCA，使用零填充
                padded = np.zeros((features.shape[0], target_dim))
                return padded.squeeze()
            
        # 如果特征维度小于目标维度，使用零填充
        elif features.shape[1] < target_dim:
            padded = np.zeros((features.shape[0], target_dim))
            padded[:, :features.shape[1]] = features
            return padded.squeeze()
            
        return features.squeeze()
        
    def fuse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """融合多模态数据
        
        Args:
            data: 包含各模态特征和元数据的字典
            
        Returns:
            融合后的特征和元数据
            
        Raises:
            ValueError: 数据格式错误或特征无效
        """
        try:
            # 提取各模态特征
            text_features = np.array(data.get('text', {}).get('features', []))
            image_features = np.array(data.get('image', {}).get('features', []))
            graph_features = np.array(data.get('graph', {}).get('features', []))
            
            # 验证特征
            if text_features.size > 0:
                text_features = self._validate_features(text_features, 'text')
            if image_features.size > 0:
                image_features = self._validate_features(image_features, 'image')
            if graph_features.size > 0:
                graph_features = self._validate_features(graph_features, 'graph')
                
            # 计算有效模态的权重
            valid_modalities = []
            if text_features.size > 0:
                valid_modalities.append(('text', text_features, self.text_weight))
            if image_features.size > 0:
                valid_modalities.append(('image', image_features, self.image_weight))
            if graph_features.size > 0:
                valid_modalities.append(('graph', graph_features, self.graph_weight))
                
            if not valid_modalities:
                raise ValueError("没有有效的模态特征")
                
            # 归一化权重
            total_weight = sum(weight for _, _, weight in valid_modalities)
            normalized_weights = [weight/total_weight for _, _, weight in valid_modalities]
            
            # 归一化特征
            normalized_features = []
            for _, features, _ in valid_modalities:
                normalized_features.append(self._normalize_features(features))
                
            # 转换特征维度
            target_dim = 512  # 使用统一的目标维度
            transformed_features = []
            for features in normalized_features:
                try:
                    transformed = self._transform_features(features, target_dim)
                    if transformed.size > 0:  # 确保转换后的特征不为空
                        transformed_features.append(transformed)
                except Exception as e:
                    logger.warning(f"特征转换失败: {str(e)}")
                    continue
                
            # 检查是否有有效的转换特征
            if not transformed_features:
                logger.warning("没有有效的转换特征，返回零向量")
                return {
                    'features': np.zeros(target_dim).tolist(),
                    'metadata': {}
                }
                
            # 加权融合
            fused_features = np.zeros_like(transformed_features[0])
            for features, weight in zip(transformed_features, normalized_weights):
                fused_features += weight * features
                
            # 合并元数据
            fused_metadata = {}
            for modality, _, _ in valid_modalities:
                if modality in data and 'metadata' in data[modality]:
                    fused_metadata[modality] = data[modality]['metadata']
                    
            return {
                'features': fused_features.tolist(),
                'metadata': fused_metadata
            }
            
        except Exception as e:
            logger.error(f"多模态融合失败: {str(e)}")
            raise 