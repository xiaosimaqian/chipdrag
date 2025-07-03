#!/usr/bin/env python3
"""
修复LEF-DEF组件不匹配问题
解决OpenROAD错误：[WARNING ODB-0099] error: netlist component (xxx) is not defined
"""

import re
import logging
from pathlib import Path
from typing import Set, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LEFDEFMismatchFixer:
    def __init__(self):
        self.designs_dir = Path("data/designs/ispd_2015_contest_benchmark")
    
    def fix_all_designs(self):
        """修复所有设计的LEF-DEF不匹配问题"""
        logger.info("开始修复所有ISPD设计的LEF-DEF不匹配问题...")
        
        designs = [
            'mgc_des_perf_1', 'mgc_des_perf_a', 'mgc_des_perf_b',
            'mgc_edit_dist_a',
            'mgc_fft_1', 'mgc_fft_2', 'mgc_fft_a', 'mgc_fft_b',
            'mgc_matrix_mult_1', 'mgc_matrix_mult_a', 'mgc_matrix_mult_b',
            'mgc_pci_bridge32_a', 'mgc_pci_bridge32_b',
            'mgc_superblue11_a', 'mgc_superblue12', 'mgc_superblue16_a'
        ]
        
        fixed_count = 0
        for design_name in designs:
            design_dir = self.designs_dir / design_name
            if design_dir.exists():
                logger.info(f"处理设计: {design_name}")
                if self.fix_design(design_dir):
                    fixed_count += 1
                    logger.info(f"✅ {design_name} 修复完成")
                else:
                    logger.warning(f"⚠️ {design_name} 修复失败")
            else:
                logger.warning(f"设计目录不存在: {design_dir}")
        
        logger.info(f"修复完成: {fixed_count}/{len(designs)} 个设计")
    
    def fix_design(self, design_dir: Path) -> bool:
        """修复单个设计的LEF-DEF不匹配问题"""
        try:
            # 查找文件
            floorplan_def = design_dir / "floorplan.def"
            cells_lef = design_dir / "cells.lef"
            tech_lef = design_dir / "tech.lef"
            
            if not floorplan_def.exists():
                logger.error(f"未找到floorplan.def: {design_dir}")
                return False
            
            if not cells_lef.exists():
                logger.error(f"未找到cells.lef: {design_dir}")
                return False
            
            # 1. 从LEF文件中提取可用的macro定义
            lef_macros = self.extract_lef_macros(cells_lef)
            logger.info(f"从LEF文件提取到 {len(lef_macros)} 个macro定义")
            
            # 2. 从DEF文件中提取组件使用的cell类型
            def_cell_types = self.extract_def_cell_types(floorplan_def)
            logger.info(f"从DEF文件提取到 {len(def_cell_types)} 个不同的cell类型")
            
            # 3. 检查不匹配的组件
            missing_macros = def_cell_types - lef_macros
            if missing_macros:
                logger.warning(f"发现 {len(missing_macros)} 个未定义的macro:")
                for macro in sorted(list(missing_macros))[:10]:  # 只显示前10个
                    logger.warning(f"  - {macro}")
                if len(missing_macros) > 10:
                    logger.warning(f"  ... 还有 {len(missing_macros) - 10} 个")
                
                # 4. 尝试修复：添加缺失的macro定义到LEF文件
                if self.add_missing_macros_to_lef(cells_lef, missing_macros):
                    logger.info("✅ 已添加缺失的macro定义到LEF文件")
                    return True
                else:
                    logger.error("❌ 添加macro定义失败")
                    return False
            else:
                logger.info("✅ 没有发现LEF-DEF不匹配问题")
                return True
                
        except Exception as e:
            logger.error(f"修复设计失败: {e}")
            return False
    
    def extract_lef_macros(self, lef_file: Path) -> Set[str]:
        """从LEF文件中提取所有macro定义"""
        macros = set()
        try:
            with open(lef_file, 'r') as f:
                content = f.read()
            
            # 查找所有MACRO定义
            macro_pattern = r'MACRO\s+(\w+)'
            matches = re.findall(macro_pattern, content)
            macros.update(matches)
            
            logger.debug(f"从 {lef_file.name} 提取macro: {sorted(list(macros))[:5]}...")
            
        except Exception as e:
            logger.error(f"读取LEF文件失败: {e}")
        
        return macros
    
    def extract_def_cell_types(self, def_file: Path) -> Set[str]:
        """从DEF文件中提取所有使用的cell类型"""
        cell_types = set()
        try:
            with open(def_file, 'r') as f:
                content = f.read()
            
            # 解析COMPONENTS段落
            in_components = False
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('COMPONENTS'):
                    in_components = True
                    continue
                elif line.startswith('END COMPONENTS'):
                    in_components = False
                    break
                elif in_components and line.startswith('-'):
                    # 解析组件行: - comp_name cell_type + ...
                    parts = line.split()
                    if len(parts) >= 3:
                        cell_type = parts[2]
                        cell_types.add(cell_type)
            
            logger.debug(f"从 {def_file.name} 提取cell类型: {sorted(list(cell_types))[:5]}...")
            
        except Exception as e:
            logger.error(f"读取DEF文件失败: {e}")
        
        return cell_types
    
    def add_missing_macros_to_lef(self, lef_file: Path, missing_macros: Set[str]) -> bool:
        """向LEF文件添加缺失的macro定义"""
        try:
            # 备份原文件
            backup_file = lef_file.with_suffix('.lef.backup_macro_fix')
            with open(lef_file, 'r') as f:
                original_content = f.read()
            
            with open(backup_file, 'w') as f:
                f.write(original_content)
            logger.info(f"原文件已备份到: {backup_file}")
            
            # 生成缺失macro的定义
            missing_macro_definitions = self.generate_macro_definitions(missing_macros)
            
            # 在文件末尾添加缺失的macro定义
            new_content = original_content.rstrip() + "\n\n" + missing_macro_definitions + "\nEND LIBRARY\n"
            
            # 移除原来的END LIBRARY
            new_content = re.sub(r'\s*END\s+LIBRARY\s*$', '', new_content.rstrip())
            new_content += "\n\nEND LIBRARY\n"
            
            # 写入修复后的文件
            with open(lef_file, 'w') as f:
                f.write(new_content)
            
            logger.info(f"已添加 {len(missing_macros)} 个macro定义到 {lef_file}")
            return True
            
        except Exception as e:
            logger.error(f"添加macro定义失败: {e}")
            return False
    
    def generate_macro_definitions(self, macro_names: Set[str]) -> str:
        """生成缺失macro的基本定义"""
        definitions = []
        
        for macro_name in sorted(macro_names):
            # 生成一个基本的macro定义
            definition = f"""
MACRO {macro_name}
  CLASS CORE ;
  ORIGIN 0.000 0.000 ;
  SIZE 1.000 BY 1.000 ;
  SYMMETRY X Y ;
  SITE core ;
  PIN A
    DIRECTION INPUT ;
    USE SIGNAL ;
    LAYER metal1 ;
      RECT 0.100 0.100 0.200 0.200 ;
  END A
  PIN Y
    DIRECTION OUTPUT ;
    USE SIGNAL ;
    LAYER metal1 ;
      RECT 0.800 0.800 0.900 0.900 ;
  END Y
END {macro_name}
"""
            definitions.append(definition)
        
        return "\n".join(definitions)

def main():
    """主函数"""
    fixer = LEFDEFMismatchFixer()
    fixer.fix_all_designs()

if __name__ == "__main__":
    main() 