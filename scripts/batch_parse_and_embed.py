import os
import json
import logging
from pathlib import Path
from modules.parsers.def_parser import parse_large_def_file, split_by_token_limit
from modules.parsers.lef_parser import parse_lef_file, split_by_token_limit as lef_split
from modules.parsers.verilog_parser import parse_verilog_file, split_by_token_limit as vlg_split

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_ollama_embedding(text, model='bge-m3:latest', base_url='http://localhost:11434'):
    import requests
    url = f'{base_url}/api/embeddings'
    payload = {"model": model, "prompt": text}
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()['embedding']

def ensure_directories(design_name: str):
    """确保所需的目录结构存在"""
    base_dir = Path("data/parsed")
    design_dir = base_dir / design_name
    
    # 创建基础目录
    base_dir.mkdir(parents=True, exist_ok=True)
    design_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (design_dir / "def").mkdir(exist_ok=True)
    (design_dir / "lef").mkdir(exist_ok=True)
    (design_dir / "verilog").mkdir(exist_ok=True)
    
    return design_dir

def get_design_files(design_root: str):
    """获取所有设计文件"""
    designs = []
    for design_dir in os.listdir(design_root):
        design_path = os.path.join(design_root, design_dir)
        if not os.path.isdir(design_path):
            continue
            
        # 查找DEF文件（可能是floorplan.def）
        def_file = next((f for f in os.listdir(design_path) if f.endswith('.def')), None)
        # 查找LEF文件（可能是cells.lef或tech.lef）
        lef_files = [f for f in os.listdir(design_path) if f.endswith('.lef')]
        # 查找Verilog文件（可能是design.v）
        verilog_file = next((f for f in os.listdir(design_path) if f.endswith('.v')), None)
        
        if any([def_file, lef_files, verilog_file]):
            designs.append({
                'name': design_dir,
                'def_path': os.path.join(design_path, def_file) if def_file else None,
                'lef_paths': [os.path.join(design_path, f) for f in lef_files] if lef_files else [],
                'verilog_path': os.path.join(design_path, verilog_file) if verilog_file else None
            })
            
    return designs

def process_and_embed(design_name: str, def_path: str = None, lef_paths: list = None, verilog_path: str = None):
    """处理并嵌入设计文件"""
    try:
        # 确保目录存在
        logger.info(f"=== 开始处理设计 {design_name} ===")
        design_dir = ensure_directories(design_name)
        logger.info(f"已创建输出目录: {design_dir}")
        
        # 处理DEF文件
        if def_path and os.path.exists(def_path):
            try:
                logger.info(f"[DEF] 开始解析文件: {def_path}")
                def_context = parse_large_def_file(def_path)
                def_output = design_dir / "def" / f"{design_name}_def.json"
                with open(def_output, 'w') as f:
                    json.dump(def_context, f, indent=2)
                logger.info(f"[DEF] 解析完成，文件大小: {os.path.getsize(def_output)} 字节")
            except Exception as e:
                logger.error(f"[DEF] 处理失败: {str(e)}")
                logger.exception("[DEF] 详细错误信息:")
        else:
            logger.warning(f"[DEF] 未找到DEF文件或文件不存在: {def_path}")
        
        # 处理LEF文件
        if lef_paths:
            try:
                logger.info(f"[LEF] 开始处理 {len(lef_paths)} 个LEF文件")
                lef_context = {}
                for i, lef_path in enumerate(lef_paths, 1):
                    if os.path.exists(lef_path):
                        logger.info(f"[LEF] 正在解析第 {i}/{len(lef_paths)} 个文件: {lef_path}")
                        lef_data = parse_lef_file(lef_path)
                        # 合并LEF数据
                        for key, value in lef_data.items():
                            if key not in lef_context:
                                lef_context[key] = value
                            elif isinstance(value, dict):
                                lef_context[key].update(value)
                            elif isinstance(value, list):
                                lef_context[key].extend(value)
                        logger.info(f"[LEF] 第 {i} 个文件解析完成")
                    else:
                        logger.warning(f"[LEF] 文件不存在: {lef_path}")
                
                if lef_context:
                    lef_output = design_dir / "lef" / f"{design_name}_lef.json"
                    with open(lef_output, 'w') as f:
                        json.dump(lef_context, f, indent=2)
                    logger.info(f"[LEF] 所有文件解析完成，输出文件大小: {os.path.getsize(lef_output)} 字节")
                else:
                    logger.warning("[LEF] 没有有效的LEF数据")
            except Exception as e:
                logger.error(f"[LEF] 处理失败: {str(e)}")
                logger.exception("[LEF] 详细错误信息:")
        else:
            logger.warning("[LEF] 未找到LEF文件")
        
        # 处理Verilog文件
        if verilog_path and os.path.exists(verilog_path):
            try:
                logger.info(f"[Verilog] 开始解析文件: {verilog_path}")
                verilog_context = parse_verilog_file(verilog_path)
                verilog_output = design_dir / "verilog" / f"{design_name}_verilog.json"
                with open(verilog_output, 'w') as f:
                    json.dump(verilog_context, f, indent=2)
                logger.info(f"[Verilog] 解析完成，文件大小: {os.path.getsize(verilog_output)} 字节")
            except Exception as e:
                logger.error(f"[Verilog] 处理失败: {str(e)}")
                logger.exception("[Verilog] 详细错误信息:")
        else:
            logger.warning(f"[Verilog] 未找到Verilog文件或文件不存在: {verilog_path}")
                
        logger.info(f"=== 设计 {design_name} 处理完成 ===\n")
                
    except Exception as e:
        logger.error(f"处理设计 {design_name} 时出错: {str(e)}")
        logger.exception("设计处理详细错误:")

def main():
    """主函数"""
    logger.info("=== 开始批量处理设计文件 ===")
    
    # 设置设计文件根目录
    design_root = "data/designs/ispd_2015_contest_benchmark"
    
    # 确保根目录存在
    if not os.path.exists(design_root):
        logger.error(f"设计文件根目录不存在: {design_root}")
        return
        
    # 获取所有设计文件
    logger.info(f"正在扫描目录: {design_root}")
    designs = get_design_files(design_root)
    if not designs:
        logger.error(f"在 {design_root} 中没有找到任何设计文件")
        return
        
    logger.info(f"找到 {len(designs)} 个设计")
    for i, design in enumerate(designs, 1):
        logger.info(f"设计 {i}/{len(designs)}: {design['name']}")
        logger.info(f"  DEF: {design['def_path']}")
        logger.info(f"  LEF: {design['lef_paths']}")
        logger.info(f"  Verilog: {design['verilog_path']}")
    
    # 处理每个设计
    for i, design in enumerate(designs, 1):
        logger.info(f"\n开始处理第 {i}/{len(designs)} 个设计")
        process_and_embed(
            design_name=design['name'],
            def_path=design['def_path'],
            lef_paths=design['lef_paths'],
            verilog_path=design['verilog_path']
        )
    
    logger.info("=== 所有设计文件处理完成 ===")

if __name__ == "__main__":
    main()