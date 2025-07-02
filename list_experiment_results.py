#!/usr/bin/env python3
"""
列出所有实验结果目录
"""

import os
from pathlib import Path
from datetime import datetime
import json

def list_experiment_results():
    """列出所有实验结果目录"""
    base_dir = Path(__file__).parent
    
    # 查找所有实验结果目录
    result_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('paper_hpwl_results_'):
            result_dirs.append(item)
    
    if not result_dirs:
        print("未找到任何实验结果目录")
        return
    
    # 按时间排序（最新的在前）
    result_dirs.sort(key=lambda x: x.name, reverse=True)
    
    print("=== 实验结果目录列表 ===\n")
    
    for i, result_dir in enumerate(result_dirs, 1):
        # 提取时间戳
        timestamp_str = result_dir.name.replace('paper_hpwl_results_', '')
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp_str
        
        print(f"{i}. {result_dir.name}")
        print(f"   时间: {formatted_time}")
        print(f"   路径: {result_dir}")
        
        # 检查是否有实验摘要
        summary_file = result_dir / "experiment_summary.md"
        if summary_file.exists():
            print(f"   📄 有实验摘要")
        
        # 检查是否有LLM日志
        llm_logs_file = result_dir / "llm_participation_logs.json"
        if llm_logs_file.exists():
            try:
                with open(llm_logs_file, 'r', encoding='utf-8') as f:
                    llm_logs = json.load(f)
                print(f"   🤖 LLM调用次数: {len(llm_logs)}")
            except:
                print(f"   🤖 有LLM日志文件")
        
        # 检查是否有可视化
        viz_dir = result_dir / "visualizations"
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.jpg"))
            print(f"   📊 可视化图表: {len(viz_files)}个")
        
        print()

def show_experiment_details(result_dir_name):
    """显示特定实验的详细信息"""
    base_dir = Path(__file__).parent
    result_dir = base_dir / result_dir_name
    
    if not result_dir.exists():
        print(f"错误: 目录 {result_dir_name} 不存在")
        return
    
    print(f"=== 实验详情: {result_dir_name} ===\n")
    
    # 显示实验摘要
    summary_file = result_dir / "experiment_summary.md"
    if summary_file.exists():
        print("📄 实验摘要:")
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print()
    
    # 显示LLM参与统计
    llm_stats_file = result_dir / "llm_participation_stats.json"
    if llm_stats_file.exists():
        try:
            with open(llm_stats_file, 'r', encoding='utf-8') as f:
                llm_stats = json.load(f)
            print("🤖 LLM参与统计:")
            print(f"  总调用次数: {llm_stats.get('total_llm_calls', 0)}")
            print(f"  参与阶段: {list(llm_stats.get('stages', {}).keys())}")
            print(f"  参与设计数: {len(llm_stats.get('designs', {}))}")
            print()
        except Exception as e:
            print(f"读取LLM统计失败: {e}")
    
    # 显示文件列表
    print("📁 结果文件:")
    for file_path in result_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(result_dir)
            size = file_path.stat().st_size
            print(f"  {rel_path} ({size} bytes)")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        # 显示特定实验详情
        result_dir_name = sys.argv[1]
        show_experiment_details(result_dir_name)
    else:
        # 列出所有实验
        list_experiment_results()
        print("\n使用方法:")
        print("  python list_experiment_results.py                    # 列出所有实验")
        print("  python list_experiment_results.py paper_hpwl_results_20250702_094000  # 查看特定实验详情")

if __name__ == "__main__":
    main() 