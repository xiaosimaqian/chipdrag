#!/usr/bin/env python3
"""
简化的OpenROAD接口测试脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_openroad_interface import EnhancedOpenROADInterface

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openroad_core():
    """测试OpenROAD核心功能"""
    try:
        # 初始化接口
        interface = EnhancedOpenROADInterface()
        
        # 设置测试文件路径
        work_dir = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1")
        verilog_file = work_dir / "design.v"
        cells_lef = work_dir / "cells.lef"
        tech_lef = work_dir / "tech.lef"
        def_file = work_dir / "mgc_des_perf_1_place.def"
        
        # 检查文件是否存在
        for file_path in [verilog_file, cells_lef, tech_lef, def_file]:
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
        
        logger.info("所有测试文件存在，开始测试OpenROAD接口")
        
        # 测试TCL脚本生成
        tcl_script = interface.create_iterative_placement_tcl(
            str(verilog_file), str(cells_lef), str(tech_lef), str(def_file),
            str(work_dir), num_iterations=3
        )
        
        logger.info("TCL脚本生成成功")
        logger.info(f"脚本长度: {len(tcl_script)} 字符")
        
        # 保存TCL脚本用于检查
        tcl_path = work_dir / "test_iterative_placement.tcl"
        with open(tcl_path, 'w') as f:
            f.write(tcl_script)
        
        logger.info(f"TCL脚本已保存到: {tcl_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openroad_core()
    if success:
        print("✅ 核心功能测试成功")
    else:
        print("❌ 核心功能测试失败")
        sys.exit(1) 