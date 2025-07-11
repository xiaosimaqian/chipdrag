"""
案例管理工具模块
提供自动补充案例的通用功能
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

def add_auto_case(design_name: str, 
                  layout_strategy: Dict, 
                  action: Dict, 
                  hpwl: float, 
                  layout_success: bool, 
                  def_file: Optional[str] = None,
                  retrieved_count: int = 0,
                  additional_info: Optional[Dict] = None) -> bool:
    """
    自动补充案例到知识库
    
    Args:
        design_name: 设计名称
        layout_strategy: 布局策略
        action: RL动作
        hpwl: HPWL值
        layout_success: 布局是否成功
        def_file: DEF文件路径
        retrieved_count: 检索到的案例数量
        additional_info: 额外信息
        
    Returns:
        bool: 是否成功添加
    """
    try:
        auto_cases_path = Path("data/knowledge_base/auto_cases.json")
        auto_cases = []
        
        # 加载现有案例
        if auto_cases_path.exists():
            try:
                with open(auto_cases_path, 'r', encoding='utf-8') as f:
                    auto_cases = json.load(f)
            except Exception as e:
                logger.warning(f"加载auto_cases.json失败: {e}")
        
        # 读取DEF文件内容
        def_content = ""
        if def_file and Path(def_file).exists():
            try:
                with open(def_file, 'r') as f:
                    def_content = f.read()
            except Exception as e:
                logger.warning(f"读取DEF文件失败: {e}")
        
        # 构造新案例
        new_case = {
            "id": int(time.time() * 1000),
            "design_name": design_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layout_strategy": layout_strategy,
            "action": action,
            "hpwl": float(hpwl) if hpwl is not None else 0.0,
            "layout_success": layout_success,
            "def_content": def_content,
            "retrieved_count": retrieved_count,
            "source": "openroad_auto"
        }
        
        # 添加额外信息
        if additional_info:
            new_case.update(additional_info)
        
        # 追加新案例
        auto_cases.append(new_case)
        
        # 保存到文件
        auto_cases_path.parent.mkdir(parents=True, exist_ok=True)
        with open(auto_cases_path, 'w', encoding='utf-8') as f:
            json.dump(auto_cases, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已自动补充案例到: {auto_cases_path} (案例ID: {new_case['id']})")
        return True
        
    except Exception as e:
        logger.error(f"自动补充案例失败: {e}")
        return False

def get_auto_cases() -> List[Dict]:
    """
    获取所有自动补充的案例
    
    Returns:
        List[Dict]: 案例列表
    """
    try:
        auto_cases_path = Path("data/knowledge_base/auto_cases.json")
        if auto_cases_path.exists():
            with open(auto_cases_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"获取自动案例失败: {e}")
        return []

def clear_auto_cases() -> bool:
    """
    清空自动补充的案例
    
    Returns:
        bool: 是否成功清空
    """
    try:
        auto_cases_path = Path("data/knowledge_base/auto_cases.json")
        if auto_cases_path.exists():
            auto_cases_path.unlink()
        logger.info("已清空自动补充案例")
        return True
    except Exception as e:
        logger.error(f"清空自动案例失败: {e}")
        return False

def get_auto_cases_stats() -> Dict[str, Any]:
    """
    获取自动案例统计信息
    
    Returns:
        Dict: 统计信息
    """
    try:
        cases = get_auto_cases()
        if not cases:
            return {"total": 0, "success_rate": 0.0, "avg_hpwl": 0.0}
        
        success_count = sum(1 for case in cases if case.get("layout_success", False))
        hpwl_values = [case.get("hpwl", 0.0) for case in cases if case.get("hpwl") is not None]
        
        return {
            "total": len(cases),
            "success_rate": success_count / len(cases) if cases else 0.0,
            "avg_hpwl": sum(hpwl_values) / len(hpwl_values) if hpwl_values else 0.0,
            "min_hpwl": min(hpwl_values) if hpwl_values else 0.0,
            "max_hpwl": max(hpwl_values) if hpwl_values else 0.0
        }
    except Exception as e:
        logger.error(f"获取自动案例统计失败: {e}")
        return {"total": 0, "success_rate": 0.0, "avg_hpwl": 0.0} 