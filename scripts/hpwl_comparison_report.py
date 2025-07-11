#!/usr/bin/env python3
"""
自动化HPWL提取与对比脚本
遍历所有设计，分别提取openroad_default.def和chipdrag_optimized.def的HPWL，计算提升率，输出JSON报告。
"""
import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiment import UnifiedPaperExperiment

def extract_hpwl(experiment, def_file):
    """优先ISPD2005解析，失败则回退原始方法"""
    if not def_file.exists():
        return None
    hpwl = experiment._extract_hpwl_from_def_ispd2005_style(def_file)
    if hpwl is None:
        hpwl = experiment._extract_hpwl_from_def(def_file)
    return hpwl

def main():
    dataset_dir = Path("dataset/ispd_2015_contest_benchmark")
    report = {}
    experiment = UnifiedPaperExperiment()

    for design_dir in dataset_dir.iterdir():
        if not design_dir.is_dir():
            continue
        design_name = design_dir.name
        def_default = design_dir / "openroad_default.def"
        def_optimized = design_dir / "chipdrag_optimized.def"
        hpwl_default = extract_hpwl(experiment, def_default)
        hpwl_optimized = extract_hpwl(experiment, def_optimized)
        improvement = None
        if hpwl_default and hpwl_optimized and hpwl_default > 0:
            improvement = ((hpwl_default - hpwl_optimized) / hpwl_default) * 100
        report[design_name] = {
            "openroad_default_hpwl": hpwl_default,
            "chipdrag_optimized_hpwl": hpwl_optimized,
            "improvement_percent": improvement
        }
        print(f"{design_name}: 默认HPWL={hpwl_default}, 优化HPWL={hpwl_optimized}, 提升率={improvement}")

    # 输出JSON报告
    out_path = Path("paper_hpwl_results/hpwl_comparison_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nHPWL对比报告已保存: {out_path}")

if __name__ == "__main__":
    main() 