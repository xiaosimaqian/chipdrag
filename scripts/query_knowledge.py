import argparse
from modules.core.rag_controller import RAGController

def query_knowledge(design_requirement_path: str):
    """基于设计需求查询多模态知识
    
    Args:
        design_requirement_path: 设计需求文本文件路径
    """
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
        # 查询知识
        results = rag_controller.query_knowledge(
            design_requirement_path=design_requirement_path,
            top_k=5
        )
        
        # 显示结果
        print(f"检索到 {len(results)} 条相关知识")
        print("知识详情:")
        
        for idx, result in enumerate(results, 1):
            try:
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0.0)
                
                print(f"\n{idx}. 来源: {metadata.get('source', 'Unknown')}")
                print(f"   类型: {metadata.get('type', 'Unknown')}")
                print(f"   文件: {metadata.get('filename', 'Unknown')}")
                print(f"   相似度: {similarity:.4f}")
                
                # 显示文本内容
                if result.get('text'):
                    text_preview = result['text'][:200] + '...' if len(result['text']) > 200 else result['text']
                    print(f"   文本预览: {text_preview}")
                
                # 显示布局信息
                if result.get('layout'):
                    print(f"   布局信息: {result['layout']}")
                
                # 显示图像信息
                if result.get('image'):
                    print(f"   图像信息: {result['image']}")
                
                # 显示结构化数据
                if result.get('structured_data'):
                    print(f"   结构化数据: {result['structured_data']}")
                    
            except Exception as e:
                print(f"   处理第 {idx} 条结果时出错: {str(e)}")
                continue
                
    except Exception as e:
        print(f"查询知识失败: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于设计需求查询多模态知识')
    parser.add_argument('design_requirement', help='设计需求文本文件路径')
    args = parser.parse_args()
    
    query_knowledge(args.design_requirement)

if __name__ == '__main__':
    main()
