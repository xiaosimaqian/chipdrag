import os
import yaml
import torch
from PIL import Image
import logging
from pathlib import Path

from chiprag.models.text_encoders.bert_encoder import BertTextEncoder
from chiprag.models.image_encoders.resnet_encoder import ResNetImageEncoder
from chiprag.models.knowledge_graph.kg_encoder import KGEncoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """加载模型配置"""
    config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_models(config):
    """下载预训练模型"""
    # 创建缓存目录
    os.makedirs(config['text_encoder']['cache_dir'], exist_ok=True)
    os.makedirs(config['image_encoder']['cache_dir'], exist_ok=True)
    os.makedirs(config['knowledge_graph']['cache_dir'], exist_ok=True)
    
    # 下载BERT模型
    logger.info("正在下载BERT模型...")
    text_encoder = BertTextEncoder(
        model_name=config['text_encoder']['model_name'],
        max_length=config['text_encoder']['max_length']
    )
    text_encoder.save_model(config['text_encoder']['model_path'])
    
    # 下载ResNet模型
    logger.info("正在下载ResNet模型...")
    image_encoder = ResNetImageEncoder(
        model_name=config['image_encoder']['model_name'],
        pretrained=config['image_encoder']['pretrained']
    )
    image_encoder.save_model(config['image_encoder']['model_path'])
    
    # 初始化知识图谱编码器
    logger.info("正在初始化知识图谱编码器...")
    kg_encoder = KGEncoder(
        num_entities=config['knowledge_graph']['num_entities'],
        num_relations=config['knowledge_graph']['num_relations'],
        embedding_dim=config['knowledge_graph']['embedding_dim']
    )
    kg_encoder.save_model(config['knowledge_graph']['model_path'])
    
    logger.info("所有模型下载完成！")

def text_encoder_example(config):
    """文本编码器使用示例"""
    logger.info("加载文本编码器...")
    text_encoder = BertTextEncoder.load_model(config['text_encoder']['model_path'])
    
    # 示例文本
    texts = [
        "这是一个时序优化的布局策略",
        "这是一个功耗优化的布局策略",
        "这是一个面积优化的布局策略"
    ]
    
    # 编码文本
    embeddings = text_encoder.encode_texts(texts)
    logger.info(f"文本编码维度: {embeddings.shape}")
    
    # 计算相似度
    similarity = text_encoder.compute_similarity(texts[0], texts[1])
    logger.info(f"文本相似度: {similarity:.4f}")

def image_encoder_example(config):
    """图像编码器使用示例"""
    logger.info("加载图像编码器...")
    image_encoder = ResNetImageEncoder.load_model(config['image_encoder']['model_path'])
    
    # 示例图像路径
    image_paths = [
        "examples/data/layout1.png",
        "examples/data/layout2.png"
    ]
    
    # 加载图像
    images = [Image.open(path) for path in image_paths]
    
    # 编码图像
    embeddings = image_encoder.encode_images(images)
    logger.info(f"图像编码维度: {embeddings.shape}")
    
    # 计算相似度
    similarity = image_encoder.compute_similarity(images[0], images[1])
    logger.info(f"图像相似度: {similarity:.4f}")

def kg_encoder_example(config):
    """知识图谱编码器使用示例"""
    logger.info("加载知识图谱编码器...")
    kg_encoder = KGEncoder.load_model(config['knowledge_graph']['model_path'])
    
    # 示例实体和关系
    entity1_id = 0
    entity2_id = 1
    relation_id = 0
    
    # 计算实体相似度
    similarity = kg_encoder.compute_entity_similarity(entity1_id, entity2_id)
    logger.info(f"实体相似度: {similarity:.4f}")
    
    # 预测尾实体
    predictions = kg_encoder.predict_tail(
        head_id=entity1_id,
        relation_id=relation_id,
        k=config['knowledge_graph']['top_k']
    )
    logger.info(f"预测的尾实体: {predictions}")

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 下载模型
    download_models(config)
    
    # 运行示例
    text_encoder_example(config)
    image_encoder_example(config)
    kg_encoder_example(config)

if __name__ == "__main__":
    main() 