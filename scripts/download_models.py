import os
import yaml
import torch
import logging
from pathlib import Path
from transformers import BertModel, BertTokenizer
import torchvision.models as models

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """加载模型配置"""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_bert_model(config):
    """下载BERT模型"""
    logger.info("正在下载BERT模型...")
    
    # 创建缓存目录
    os.makedirs(config['text_encoder']['cache_dir'], exist_ok=True)
    
    # 下载模型和分词器
    model = BertModel.from_pretrained(
        config['text_encoder']['model_name'],
        cache_dir=config['text_encoder']['cache_dir']
    )
    tokenizer = BertTokenizer.from_pretrained(
        config['text_encoder']['model_name'],
        cache_dir=config['text_encoder']['cache_dir']
    )
    
    # 保存模型
    model_path = Path(config['text_encoder']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'max_length': config['text_encoder']['max_length']
    }, model_path)
    
    logger.info(f"BERT模型已保存到: {model_path}")

def download_resnet_model(config):
    """下载ResNet模型"""
    logger.info("正在下载ResNet模型...")
    
    # 创建缓存目录
    os.makedirs(config['image_encoder']['cache_dir'], exist_ok=True)
    
    # 下载预训练模型
    model = models.resnet50(pretrained=True)
    
    # 保存模型
    model_path = Path(config['image_encoder']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'transform': {
            'image_size': config['image_encoder']['image_size'],
            'mean': config['image_encoder']['mean'],
            'std': config['image_encoder']['std']
        }
    }, model_path)
    
    logger.info(f"ResNet模型已保存到: {model_path}")

def initialize_kg_encoder(config):
    """初始化知识图谱编码器"""
    logger.info("正在初始化知识图谱编码器...")
    
    # 创建缓存目录
    os.makedirs(config['knowledge_graph']['cache_dir'], exist_ok=True)
    
    # 初始化模型
    from chiprag.models.knowledge_graph.kg_encoder import KGEncoder
    model = KGEncoder(
        num_entities=config['knowledge_graph']['num_entities'],
        num_relations=config['knowledge_graph']['num_relations'],
        embedding_dim=config['knowledge_graph']['embedding_dim']
    )
    
    # 保存模型
    model_path = Path(config['knowledge_graph']['model_path'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(model_path)
    
    logger.info(f"知识图谱编码器已保存到: {model_path}")

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 下载模型
    download_bert_model(config)
    download_resnet_model(config)
    initialize_kg_encoder(config)
    
    logger.info("所有模型下载完成！")

if __name__ == "__main__":
    main() 