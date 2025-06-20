#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from run_rag import RAGController

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 1. 创建控制器
    controller = RAGController()
    
    # 2. 定义设计规格
    design_spec = {
        "name": "example_design",
        "area": {
            "width": 2000,
            "height": 2000
        },
        "die_area": {
            "width": 2000,
            "height": 2000
        },
        "components": [
            {
                "name": "cpu_core",
                "type": "macro",
                "width": 400,
                "height": 400
            },
            {
                "name": "gpu_core",
                "type": "macro",
                "width": 500,
                "height": 500
            },
            {
                "name": "memory_controller",
                "type": "macro",
                "width": 300,
                "height": 200
            },
            {
                "name": "io_controller",
                "type": "macro",
                "width": 200,
                "height": 300
            }
        ],
        "nets": [
            {
                "name": "cpu_gpu_net",
                "pins": [
                    {"component": "cpu_core", "x": 200, "y": 200},
                    {"component": "gpu_core", "x": 250, "y": 250}
                ]
            },
            {
                "name": "cpu_mem_net",
                "pins": [
                    {"component": "cpu_core", "x": 300, "y": 300},
                    {"component": "memory_controller", "x": 150, "y": 100}
                ]
            },
            {
                "name": "gpu_mem_net",
                "pins": [
                    {"component": "gpu_core", "x": 350, "y": 350},
                    {"component": "memory_controller", "x": 200, "y": 150}
                ]
            },
            {
                "name": "io_net",
                "pins": [
                    {"component": "io_controller", "x": 100, "y": 150},
                    {"component": "memory_controller", "x": 250, "y": 100}
                ]
            }
        ],
        "constraints": {
            "max_wirelength": 2000,
            "max_congestion": 0.8,
            "min_spacing": 50,
            "power_budget": 100
        }
    }
    
    try:
        # 3. 检索和增强知识
        logger.info("开始检索和增强知识...")
        enhanced_knowledge = controller.rag_system.retrieve_and_enhance(
            design_input=design_spec,
            constraints=design_spec["constraints"]
        )
        logger.info("知识检索和增强完成")
        
        # 4. 生成布局
        logger.info("开始生成布局...")
        layout = controller.layout_generator.generate_layout(
            design_info=design_spec,
            hierarchy_info={},
            knowledge_base=enhanced_knowledge
        )
        logger.info("布局生成完成")
        
        # 5. 评估布局
        logger.info("开始评估布局...")
        evaluation = controller.evaluator.evaluate(layout)
        logger.info("布局评估完成")
        
        # 6. 保存结果
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存增强后的知识
        with open(output_dir / "enhanced_knowledge.json", "w", encoding="utf-8") as f:
            json.dump(enhanced_knowledge, f, indent=2, ensure_ascii=False)
        
        # 保存布局结果
        with open(output_dir / "layout.json", "w", encoding="utf-8") as f:
            json.dump(layout, f, indent=2, ensure_ascii=False)
        
        # 保存评估结果
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {output_dir} 目录")
        
        # 7. 打印关键指标
        print("\n=== 布局评估结果 ===")
        print(f"线长: {evaluation.get('wirelength', 'N/A')}")
        print(f"拥塞: {evaluation.get('congestion', 'N/A')}")
        print(f"时序: {evaluation.get('timing', 'N/A')}")
        
        print("\n=== 组件位置 ===")
        for comp in layout["components"]:
            print(f"{comp['name']}: ({comp['x']}, {comp['y']})")
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 