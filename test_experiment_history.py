#!/usr/bin/env python3
"""
测试实验历史列表功能
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from paper_hpwl_comparison_experiment import PaperHPWLComparisonExperiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_experiment_history():
    """测试实验历史列表功能"""
    logger.info("=== 测试实验历史列表功能 ===")
    
    # 创建实验实例
    experiment = PaperHPWLComparisonExperiment()
    
    # 测试历史实验列表功能
    experiment._list_all_experiment_results()
    
    logger.info("测试完成")

if __name__ == "__main__":
    test_experiment_history() 