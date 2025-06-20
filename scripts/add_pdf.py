# chiprag/scripts/add_pdf.py

import os
import sys
from pathlib import Path
import glob
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from modules.core.rag_controller import RAGController

def add_pdf_to_rag(pdf_path: str, design_requirement_path: str):
    """将PDF添加到RAG系统并验证相关知识
    
    Args:
        pdf_path: PDF文件的路径
        design_requirement_path: 设计需求文本文件路径
    """
    # 读取设计需求
    try:
        with open(design_requirement_path, 'r', encoding='utf-8') as f:
            design_requirement = f.read().strip()
        print(f"设计需求: {design_requirement}\n")
    except Exception as e:
        print(f"读取设计需求失败: {str(e)}")
        return
    
    # 初始化RAG控制器
    config = {
        'knowledge_base': {
            'text_path': 'data/knowledge_base/text',
            'image_path': 'data/knowledge_base/images',
            'structured_path': 'data/knowledge_base/structured',
            'graph_path': 'data/knowledge_base/graph',
            'layout_experience': 'data/knowledge_base/layout_experience'
        },
        'embedding': {
            'model_name': 'bge-m3',
            'api_base': 'http://localhost:11434'
        },
        'cache_dir': 'data/cache'
    }
    
    rag_controller = RAGController(config)
    
    try:
        # 添加PDF
        success = rag_controller.add_pdf_knowledge(pdf_path)
        
        if success:
            print(f"成功将PDF添加到知识库: {pdf_path}")
            
            # 验证添加结果
            query_results = rag_controller.verify_knowledge(
                query={
                    'text': design_requirement,
                    'type': 'layout_design',
                    'source': 'pdf',
                    'filename': Path(pdf_path).name
                },
                top_k=5
            )
            
            # 显示更详细的信息
            print(f"检索到 {len(query_results)} 条相关知识")
            print("知识详情:")
            
            for idx, result in enumerate(query_results, 1):
                try:
                    # 获取文本内容
                    text = result.get('text', '')
                    metadata = result.get('metadata', {})
                    similarity = result.get('similarity', 0.0)
                    
                    # 显示文本预览
                    text_preview = text[:200] + '...' if len(text) > 200 else text
                    
                    print(f"\n{idx}. 文件: {metadata.get('filename', 'Unknown')}")
                    print(f"   相似度: {similarity:.4f}")
                    print(f"   内容预览: {text_preview}")
                    
                except Exception as e:
                    print(f"   处理第 {idx} 条结果时出错: {str(e)}")
                    continue
                    
        else:
            print(f"添加PDF失败: {pdf_path}")
            
    except Exception as e:
        print(f"处理PDF {pdf_path} 失败: {str(e)}")
        print(f"添加PDF失败: {pdf_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='添加PDF到RAG系统')
    parser.add_argument('pdf_path', help='PDF文件路径或模式')
    parser.add_argument('--design-requirement', '-d', required=True,
                      help='设计需求文本文件路径')
    args = parser.parse_args()
    
    # 获取所有匹配的PDF文件
    pdf_files = glob.glob(args.pdf_path)
    print(f"找到 {len(pdf_files)} 个PDF文件\n")
    
    # 处理每个PDF文件
    for pdf_file in pdf_files:
        print(f"处理文件: {pdf_file}")
        add_pdf_to_rag(pdf_file, args.design_requirement)

if __name__ == '__main__':
    main()