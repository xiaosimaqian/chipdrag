#!/usr/bin/env python3
"""
检查知识库数据状态
"""

import os
import pickle
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_knowledge_base():
    """检查知识库数据状态"""
    
    # 检查数据目录
    data_dir = Path("data/knowledge_base")
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    logger.info(f"检查数据目录: {data_dir}")
    
    # 检查cases.pkl
    cases_file = data_dir / "cases.pkl"
    if cases_file.exists():
        try:
            with open(cases_file, 'rb') as f:
                cases = pickle.load(f)
            logger.info(f"cases.pkl 存在，包含 {len(cases)} 个案例")
            
            # 显示前几个案例的信息
            for i, case in enumerate(cases[:3]):
                logger.info(f"案例 {i}: {case.get('id', 'unknown')} - {case.get('name', 'unnamed')}")
                
        except Exception as e:
            logger.error(f"读取cases.pkl失败: {str(e)}")
    else:
        logger.warning("cases.pkl 不存在")
    
    # 检查knowledge_graph.pkl
    graph_file = data_dir / "knowledge_graph.pkl"
    if graph_file.exists():
        try:
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"knowledge_graph.pkl 存在，大小: {os.path.getsize(graph_file)} bytes")
        except Exception as e:
            logger.error(f"读取knowledge_graph.pkl失败: {str(e)}")
    else:
        logger.warning("knowledge_graph.pkl 不存在")
    
    # 检查其他文件
    other_files = [
        "components.json",
        "triples.json", 
        "relations.json",
        "entities.json"
    ]
    
    for filename in other_files:
        file_path = data_dir / filename
        if file_path.exists():
            size = os.path.getsize(file_path)
            logger.info(f"{filename} 存在，大小: {size} bytes")
        else:
            logger.warning(f"{filename} 不存在")

def test_knowledge_base_loading():
    """测试知识库加载"""
    try:
        from modules.knowledge.knowledge_base import KnowledgeBase
        
        # 加载配置
        config = {
            "path": "data/knowledge_base",
            "format": "pkl",
            "layout_experience": "data/knowledge_base"
        }
        
        # 初始化知识库
        kb = KnowledgeBase(config)
        
        logger.info(f"知识库初始化成功，包含 {len(kb.cases)} 个案例")
        
        # 测试相似案例检索
        test_query = {
            "name": "test_design",
            "components": [],
            "nets": []
        }
        
        similar_cases = kb.get_similar_cases(test_query, top_k=3)
        logger.info(f"相似案例检索返回 {len(similar_cases)} 个结果")
        
        return True
        
    except Exception as e:
        logger.error(f"知识库加载测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("开始检查知识库数据状态...")
    
    check_knowledge_base()
    
    logger.info("开始测试知识库加载...")
    test_knowledge_base_loading()
    
    logger.info("检查完成") 