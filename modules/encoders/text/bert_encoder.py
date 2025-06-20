"""
BERT文本编码器模块
"""

import os
import json
import logging
import torch
import numpy as np
import ssl
import certifi
import requests
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModel
from modules.encoders.base_encoder import BaseEncoder

def get_system_config_path():
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../configs/system.json'))
    if os.path.exists(abs_path):
        return abs_path
    alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs/system.json'))
    if os.path.exists(alt_path):
        return alt_path
    raise FileNotFoundError(f"未找到系统配置文件，建议放在: {abs_path}")

class BertTextEncoder(BaseEncoder):
    """BERT文本编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化BERT文本编码器
        
        Args:
            config: 配置参数字典
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化BERT文本编码器")
        super().__init__(config)
        
        # 初始化模型
        self._init_model()
        
    def _init_model(self):
        """初始化模型"""
        try:
            # 获取配置
            self.model_path = self.config.get('model_path', 'bert-base-chinese')
            self.max_length = self.config.get('max_length', 512)
            self.batch_size = self.config.get('batch_size', 32)
        
            # 读取系统配置
            system_config_path = get_system_config_path()
            with open(system_config_path, 'r') as f:
                system_config = json.load(f)
                self.model_path = system_config.get('text_encoder', {}).get('model_path', self.model_path)
            
            # 初始化模型
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                self.logger.info(f"BERT模型加载成功: {self.model_path}")
            except Exception as e:
                self.logger.error(f"BERT模型加载失败: {str(e)}")
                raise
            
            self.model_name = self.config.get('model_name', 'bert-base-chinese')
            self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
            self.use_ollama = self.config.get('use_ollama', False)
            self.ollama_model = self.config.get('ollama_model', 'llama2')
            
            # 设置SSL验证
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            
            if self.use_ollama:
                self._init_ollama()
            else:
                self._init_local_model()
                
        except Exception as e:
            self.logger.error(f"初始化模型失败: {str(e)}")
            raise
        
    def _init_ollama(self):
        """初始化Ollama模型"""
        try:
            if not self.use_ollama:
                return
            
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                self.logger.warning(f"无法连接到Ollama服务: {self.ollama_url}")
                self.use_ollama = False
                return
            
            available_models = [model['name'] for model in response.json()['models']]
            if self.ollama_model not in available_models:
                self.logger.warning(f"模型 {self.ollama_model} 不可用，可用模型: {available_models}")
                self.use_ollama = False
                return
            
            self.logger.info(f"成功初始化Ollama模型: {self.ollama_model}")
        except Exception as e:
            self.logger.error(f"初始化Ollama模型失败: {str(e)}")
            self.use_ollama = False
            
    def _init_local_model(self):
        """初始化本地模型"""
        try:
            if not os.path.exists(self.model_path):
                # 如果模型路径不存在，尝试从Hugging Face下载
                from transformers import AutoTokenizer, AutoModel
                
                self.logger.info(f"从Hugging Face下载模型到 {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                self.model = AutoModel.from_pretrained("bert-base-chinese")
                
                # 保存模型到本地
                os.makedirs(self.model_path, exist_ok=True)
                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path)
            else:
                # 加载本地模型
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
                
            # 将模型移到指定设备
            self.model = self.model.to(self.device)
            self.logger.info(f"成功加载本地模型: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"初始化本地模型失败: {str(e)}")
            raise
            
    def _validate_text(self, text: str) -> None:
        """验证输入文本
        
        Args:
            text: 输入文本
            
        Raises:
            ValueError: 文本无效
        """
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串")
        if not text.strip():
            raise ValueError("输入文本不能为空")
            
    def _validate_texts(self, texts: List[str]) -> None:
        """验证输入文本列表
        
        Args:
            texts: 输入文本列表
            
        Raises:
            ValueError: 文本列表无效
        """
        if not isinstance(texts, list):
            raise ValueError("输入必须是列表")
        if not texts:
            raise ValueError("输入列表不能为空")
        for text in texts:
            self._validate_text(text)
            
    def _process_batch(self, texts: List[str], start_idx: int) -> torch.Tensor:
        """处理一批文本
        
        Args:
            texts: 文本列表
            start_idx: 起始索引
            
        Returns:
            文本编码向量
        """
        end_idx = min(start_idx + self.batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # 对文本进行编码
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将输入移到模型所在设备
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # 使用[CLS]标记的输出作为文本表示
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 清理内存
        del encoded, input_ids, attention_mask, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return text_embeddings
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            texts: 文本列表
            
        Returns:
            文本编码向量
        """
        self._validate_texts(texts)
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_embeddings = self._process_batch(texts, i)
            all_embeddings.append(batch_embeddings)
            
        # 合并所有批次的编码
        return torch.cat(all_embeddings, dim=0)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本编码向量
        """
        self._validate_text(text)
        return self.forward([text])[0]
        
    def encode(self, text: Union[str, List[str], Dict]) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 输入文本、文本列表或包含文本的字典
            
        Returns:
            torch.Tensor: 文本编码向量
        """
        try:
            # 处理不同类型的输入
            if isinstance(text, dict):
                # 如果是字典，尝试提取文本内容
                if 'text' in text:
                    text = text['text']
                elif 'content' in text:
                    text = text['content']
                else:
                    # 如果找不到文本字段，使用字典的字符串表示
                    text = str(text)
            elif isinstance(text, list):
                # 如果是列表，取第一个元素
                if text:
                    text = text[0] if isinstance(text[0], str) else str(text[0])
                else:
                    text = ""
            
            # 确保text是字符串
            if not isinstance(text, str):
                text = str(text)
            
            # 预处理文本
            text = self._preprocess_text(text)
            
            if self.use_ollama:
                return self._encode_with_ollama(text)
            else:
                return self._encode_with_local_model(text)
                
        except Exception as e:
            self.logger.error(f"文本编码失败: {str(e)}")
            return torch.zeros(768)  # 返回零向量
            
    def _encode_with_ollama(self, text: str) -> torch.Tensor:
        """使用Ollama编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码
        """
        try:
            # 调用Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.ollama_model,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ollama API调用失败: {response.text}")
                
            # 获取嵌入向量
            embedding = torch.tensor(response.json().get('embedding', []), device=self.device)
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Ollama编码失败: {str(e)}")
            return torch.zeros(768)
            
    def _encode_with_local_model(self, text: str) -> torch.Tensor:
        """使用本地模型编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码
        """
        try:
            # 对文本进行编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # 使用[CLS]标记的输出作为文本表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            return embeddings.mean(dim=0)  # 返回第一个句子的编码
            
        except Exception as e:
            self.logger.error(f"本地模型编码失败: {str(e)}")
            return torch.zeros(768)
            
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            str: 预处理后的文本
        """
        # 这里可以添加文本预处理步骤
        # 例如：去除特殊字符、标准化等
        return text.strip()
        
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本编码向量矩阵
        """
        return self.forward(texts)
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度得分
        """
        try:
            # 编码文本
            embedding1 = self.encode(text1)
            embedding2 = self.encode(text2)
            
            # 确保张量维度正确
            if embedding1.dim() == 1:
                embedding1 = embedding1.unsqueeze(0)
            if embedding2.dim() == 1:
                embedding2 = embedding2.unsqueeze(0)
        
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                embedding1,
                embedding2,
                dim=1  # 在特征维度上计算相似度
            )
            
            return similarity.item()
            
        except Exception as e:
            self.logger.error(f"计算相似度失败: {str(e)}")
            return 0.5  # 返回默认相似度
        
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        try:
            if not self.use_ollama:
                torch.save(self.model, path)
                self.logger.info(f"模型已保存到: {path}")
                
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            raise
        
    @classmethod
    def load_model(cls, path: str) -> 'BertTextEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.tokenizer = checkpoint['tokenizer']
        model.max_length = checkpoint['max_length']
        model.batch_size = checkpoint['batch_size']
        model.logger.info(f"模型已从 {path} 加载")
        return model 

    def preprocess(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """预处理文本数据
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            Dict[str, torch.Tensor]: 预处理后的数据
        """
        if isinstance(text, str):
            text = [text]
            
        # 对文本进行编码
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将数据移动到指定设备
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        """批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            torch.Tensor: 编码后的向量
        """
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        return torch.cat(all_embeddings, dim=0) 