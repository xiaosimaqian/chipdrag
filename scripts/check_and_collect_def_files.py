#!/usr/bin/env python3
"""
自动化批量收集和校验ISPD所有DEF文件
检查每个设计目录下的模块名.def文件是否存在、大小、行数等
"""

import os
import json
import re
from pathlib import Path

# 配置
ISPD_DIR = "data/designs/ispd_2015_contest_benchmark"
BENCHMARKS = [d for d in os.listdir(ISPD_DIR) if os.path.isdir(os.path.join(ISPD_DIR, d))]

def validate_def_content(content):
    """简单校验DEF文件内容"""
    lines = content.split('\n')
    
    # 检查关键段
    has_placement = any("PLACEMENT" in line for line in lines)
    has_end_design = any("END DESIGN" in line for line in lines)
    has_components = any("COMPONENTS" in line for line in lines)
    has_nets = any("NETS" in line for line in lines)
    
    # 检查PLACEMENT段是否有实际内容
    placement_start = -1
    placement_end = -1
    for i, line in enumerate(lines):
        if "PLACEMENT" in line:
            placement_start = i
        elif placement_start != -1 and ";" in line and "PLACEMENT" not in line:
            placement_end = i
            break
    
    has_placement_content = False
    if placement_start != -1 and placement_end != -1:
        placement_lines = lines[placement_start:placement_end+1]
        # 检查是否有实际的placement语句（不是空的）
        has_placement_content = any(
            re.match(r'\s*\w+\s+\w+\s+\+?\s*PLACED\s+\(\s*\d+\s+\d+\s*\)', line)
            for line in placement_lines
        )
    
    return {
        "has_placement": has_placement,
        "has_end_design": has_end_design,
        "has_components": has_components,
        "has_nets": has_nets,
        "has_placement_content": has_placement_content,
        "valid": has_placement and has_end_design and has_placement_content
    }

def main():
    print("=" * 60)
    print("ISPD 2015 DEF文件收集与校验报告")
    print("=" * 60)
    
    report = {}
    total_exists = 0
    total_valid = 0
    
    for design in sorted(BENCHMARKS):
        design_dir = os.path.join(ISPD_DIR, design)
        def_file = os.path.join(design_dir, f"{design}.def")
        
        result = {
            "exists": False,
            "size": 0,
            "lines": 0,
            "valid": False,
            "validation_details": {},
            "message": ""
        }
        
        if os.path.exists(def_file):
            result["exists"] = True
            result["size"] = os.path.getsize(def_file)
            total_exists += 1
            
            try:
                with open(def_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                result["lines"] = len(lines)
                
                # 校验DEF文件内容
                validation = validate_def_content(content)
                result["validation_details"] = validation
                result["valid"] = validation["valid"]
                
                if result["valid"]:
                    total_valid += 1
                    result["message"] = "DEF文件存在且格式正常"
                else:
                    missing = []
                    if not validation["has_placement"]:
                        missing.append("PLACEMENT段")
                    if not validation["has_end_design"]:
                        missing.append("END DESIGN")
                    if not validation["has_placement_content"]:
                        missing.append("实际placement内容")
                    result["message"] = f"DEF文件存在但缺少: {', '.join(missing)}"
                    
            except Exception as e:
                result["message"] = f"读取DEF文件异常: {e}"
        else:
            result["message"] = "DEF文件不存在"
        
        report[design] = result
        
        # 打印每个设计的状态
        status = "✅" if result["valid"] else "⚠️" if result["exists"] else "❌"
        print(f"{status} {design}: {result['message']}")
        if result["exists"]:
            print(f"   大小: {result['size']:,} bytes, 行数: {result['lines']:,}")
    
    # 输出汇总报告
    print("\n" + "=" * 60)
    print("汇总报告")
    print("=" * 60)
    print(f"总设计数: {len(BENCHMARKS)}")
    print(f"DEF文件存在: {total_exists}")
    print(f"DEF文件有效: {total_valid}")
    print(f"存在率: {total_exists/len(BENCHMARKS)*100:.1f}%")
    print(f"有效率: {total_valid/len(BENCHMARKS)*100:.1f}%")
    
    # 保存详细报告
    report_path = os.path.join(ISPD_DIR, "def_check_report.json")
    with open(report_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细报告已保存到: {report_path}")
    
    # 列出所有DEF文件路径
    print("\n所有DEF文件路径:")
    for design in sorted(BENCHMARKS):
        def_file = os.path.join(ISPD_DIR, design, f"{design}.def")
        if os.path.exists(def_file):
            print(f"  {def_file}")

if __name__ == "__main__":
    main() 