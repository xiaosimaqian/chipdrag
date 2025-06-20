import re
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def split_by_token_limit(content: str, max_tokens: int = 1000) -> List[str]:
    """将DEF文件内容按token数量分割
    
    Args:
        content: DEF文件内容
        max_tokens: 每个分片的最大token数量
        
    Returns:
        分割后的内容列表
    """
    # 按分号分割语句
    statements = re.split(r';\s*', content)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for stmt in statements:
        # 简单估算token数量（按空格分割）
        stmt_tokens = len(stmt.split())
        
        if current_tokens + stmt_tokens > max_tokens and current_chunk:
            chunks.append(';\n'.join(current_chunk) + ';')
            current_chunk = []
            current_tokens = 0
            
        current_chunk.append(stmt)
        current_tokens += stmt_tokens
        
    if current_chunk:
        chunks.append(';\n'.join(current_chunk) + ';')
        
    return chunks

def parse_large_def_file(def_file: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """解析大型DEF文件
    
    Args:
        def_file: DEF文件路径
        max_tokens: 每个分片的最大token数量
        
    Returns:
        解析后的数据字典
    """
    logger.info(f"开始解析大型DEF文件: {def_file}")
    
    try:
        with open(def_file, 'r') as f:
            content = f.read()
            
        # 分割文件内容，保持DEF语句的完整性
        statements = []
        current_statement = []
        current_tokens = 0
        in_block = False
        block_name = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # 计算当前行的token数量
            line_tokens = len(line.split())
            
            # 检查是否进入或离开一个块
            block_start = re.match(r'(\w+)\s+(\d+)', line)
            if block_start:
                if in_block:
                    # 结束当前块
                    if current_statement:
                        statements.append(' '.join(current_statement))
                        current_statement = []
                        current_tokens = 0
                in_block = True
                block_name = block_start.group(1)
                current_statement = [line]
                current_tokens = line_tokens
                continue
                
            # 如果当前行包含分号，说明是一个完整的语句
            if ';' in line:
                parts = line.split(';')
                for i, part in enumerate(parts[:-1]):
                    if current_statement:
                        current_statement.append(part)
                        statements.append(' '.join(current_statement) + ';')
                        current_statement = []
                        current_tokens = 0
                    else:
                        statements.append(part + ';')
                if parts[-1].strip():
                    current_statement = [parts[-1]]
                    current_tokens = line_tokens
            else:
                if current_tokens + line_tokens > max_tokens and current_statement:
                    statements.append(' '.join(current_statement))
                    current_statement = []
                    current_tokens = 0
                current_statement.append(line)
                current_tokens += line_tokens
                
        if current_statement:
            statements.append(' '.join(current_statement))
            
        logger.info(f"DEF文件已分割为 {len(statements)} 个语句")
    
        # 解析每个语句
        result = {
            'VERSION': None,
            'DESIGN': None,
            'UNITS': None,
            'DIEAREA': None,
            'ROWS': [],
            'TRACKS': [],
            'GCELLGRID': [],
            'VIAS': [],
            'NONDEFAULTRULES': [],
            'REGIONS': [],
            'COMPONENTS': [],
            'PINS': [],
            'NETS': [],
            'SPECIALNETS': []
        }
        
        # 首先解析头部信息（VERSION, DESIGN, UNITS等）
        header_parser = DEFParser()
        for statement in statements:
            if any(keyword in statement for keyword in ['VERSION', 'DESIGN', 'UNITS', 'DIEAREA']):
                try:
                    chunk_data = header_parser.parse_def(statement)
                    for key, value in chunk_data.items():
                        if value is not None:
                            result[key] = value
                except Exception as e:
                    logger.error(f"解析头部信息时出错: {str(e)}")
                    continue
        
        # 然后解析其他部分
        for i, statement in enumerate(statements):
            if any(keyword in statement for keyword in ['VERSION', 'DESIGN', 'UNITS', 'DIEAREA']):
                continue
                
            logger.info(f"正在解析第 {i+1}/{len(statements)} 个语句")
            try:
                parser = DEFParser()
                chunk_data = parser.parse_def(statement)
                # 合并结果
                for key, value in chunk_data.items():
                    if value is None:
                        continue
                    if key not in result:
                        result[key] = value
                    elif isinstance(value, dict):
                        if result[key] is None:
                            result[key] = {}
                        result[key].update(value)
                    elif isinstance(value, list):
                        if result[key] is None:
                            result[key] = []
                        result[key].extend(value)
            except Exception as e:
                logger.error(f"解析第 {i+1} 个语句时出错: {str(e)}")
                continue
                
        logger.info("DEF文件解析完成")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"DEF文件不存在: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"解析DEF文件时出错: {str(e)}")
        return None
            
# 为向后兼容性提供别名
parse_def_file = parse_large_def_file

class DEFParser:
    def __init__(self):
        self.data = {
            'VERSION': None,
            'DESIGN': None,
            'UNITS': None,
            'DIEAREA': None,
            'ROWS': [],
            'TRACKS': [],
            'GCELLGRID': [],
            'VIAS': [],
            'NONDEFAULTRULES': [],
            'REGIONS': [],
            'COMPONENTS': [],
            'PINS': [],
            'NETS': [],
            'SPECIALNETS': []
        }
        
    def parse_def(self, content: str) -> Dict[str, Any]:
        """解析DEF文件内容
        
        Args:
            content: DEF文件内容
            
        Returns:
            解析后的数据字典
        """
        logger.info("开始解析DEF文件内容")
            
        try:
            # 解析版本
            version_match = re.search(r'VERSION\s+(\S+)', content)
            if version_match:
                self.data['VERSION'] = version_match.group(1)
                logger.info(f"解析到版本: {self.data['VERSION']}")
                
            # 解析设计名称
            design_match = re.search(r'DESIGN\s+(\S+)', content)
            if design_match:
                self.data['DESIGN'] = design_match.group(1)
                logger.info(f"解析到设计名称: {self.data['DESIGN']}")
            
            # 解析单位
            units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
            if units_match:
                self.data['UNITS'] = {
                    'distance': int(units_match.group(1)),
                    'unit': 'MICRONS'
                }
                logger.info(f"解析到单位: {self.data['UNITS']}")
            
            # 解析芯片区域
            diearea_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
            if diearea_match:
                self.data['DIEAREA'] = {
                    'x1': int(diearea_match.group(1)),
                    'y1': int(diearea_match.group(2)),
                    'x2': int(diearea_match.group(3)),
                    'y2': int(diearea_match.group(4))
                }
                logger.info("解析到芯片区域")
            
            # 解析行
            row_pattern = r'ROW\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\w+)\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+)'
            row_matches = re.finditer(row_pattern, content)
            
            for row_match in row_matches:
                row = {
                    'name': row_match.group(1),
                    'site': row_match.group(2),
                    'x': int(row_match.group(3)),
                    'y': int(row_match.group(4)),
                    'orientation': row_match.group(5),
                    'num_x': int(row_match.group(6)),
                    'num_y': int(row_match.group(7)),
                    'step_x': int(row_match.group(8)),
                    'step_y': int(row_match.group(9))
                    }
                self.data['ROWS'].append(row)
                logger.info(f"解析到行: {row['name']}")
            
            # 解析轨道
            track_pattern = r'TRACKS\s+(\w+)\s+(\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\w+)'
            track_matches = re.finditer(track_pattern, content)
            
            for track_match in track_matches:
                track = {
                    'direction': track_match.group(1),
                    'start': int(track_match.group(2)),
                    'num': int(track_match.group(3)),
                    'step': int(track_match.group(4)),
                    'layer': track_match.group(5)
                }
                self.data['TRACKS'].append(track)
                logger.info(f"解析到轨道: {track['direction']} {track['layer']}")
            
            # 解析网格
            grid_pattern = r'GCELLGRID\s+(\w+)\s+(\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)'
            grid_matches = re.finditer(grid_pattern, content)
            
            for grid_match in grid_matches:
                grid = {
                    'direction': grid_match.group(1),
                    'start': int(grid_match.group(2)),
                    'num': int(grid_match.group(3)),
                    'step': int(grid_match.group(4))
                    }
                self.data['GCELLGRID'].append(grid)
                logger.info(f"解析到网格: {grid['direction']}")
            
            # 解析通孔
            via_pattern = r'VIA\s+(\d+)\s+(\w+)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)'
            via_matches = re.finditer(via_pattern, content)
            
            for via_match in via_matches:
                via = {
                    'id': int(via_match.group(1)),
                    'name': via_match.group(2),
                    'x': int(via_match.group(3)),
                    'y': int(via_match.group(4)),
                    'layer': via_match.group(5)
                    }
                self.data['VIAS'].append(via)
                logger.info(f"解析到通孔: {via['name']}")
            
            # 解析非默认规则
            ndr_pattern = r'NONDEFAULTRULE\s+(\w+)\s+(\w+)'
            ndr_matches = re.finditer(ndr_pattern, content)
            
            for ndr_match in ndr_matches:
                ndr = {
                    'name': ndr_match.group(1),
                    'type': ndr_match.group(2)
                }
                self.data['NONDEFAULTRULES'].append(ndr)
                logger.info(f"解析到非默认规则: {ndr['name']}")
            
            # 解析区域
            region_pattern = r'REGION\s+(\w+)\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)'
            region_matches = re.finditer(region_pattern, content)
                
            for region_match in region_matches:
                region = {
                    'name': region_match.group(1),
                    'x1': int(region_match.group(2)),
                    'y1': int(region_match.group(3)),
                    'x2': int(region_match.group(4)),
                    'y2': int(region_match.group(5))
                }
                self.data['REGIONS'].append(region)
                logger.info(f"解析到区域: {region['name']}")
            
            # 解析组件
            comp_pattern = r'COMPONENTS\s+(\d+)\s*;\s*(.*?)(?=END\s+COMPONENTS|$)'
            comp_section = re.search(comp_pattern, content, re.DOTALL)
            if comp_section:
                comp_content = comp_section.group(2)
                comp_item_pattern = r'-?\s*(\w+)\s+(\w+)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)'
                comp_matches = re.finditer(comp_item_pattern, comp_content)
            
                for comp_match in comp_matches:
                    comp = {
                        'name': comp_match.group(1),
                        'type': comp_match.group(2),
                        'x': int(comp_match.group(3)),
                        'y': int(comp_match.group(4)),
                        'orientation': comp_match.group(5)
                    }
                    self.data['COMPONENTS'].append(comp)
                    logger.info(f"解析到组件: {comp['name']}")
            
            # 解析引脚
            pin_pattern = r'PINS\s+(\d+)\s*;\s*(.*?)(?=END\s+PINS|$)'
            pin_section = re.search(pin_pattern, content, re.DOTALL)
            if pin_section:
                pin_content = pin_section.group(2)
                pin_item_pattern = r'-?\s*(\w+)\s+(\w+)\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)'
                pin_matches = re.finditer(pin_item_pattern, pin_content)
            
                for pin_match in pin_matches:
                    pin = {
                        'name': pin_match.group(1),
                        'type': pin_match.group(2),
                        'x': int(pin_match.group(3)),
                        'y': int(pin_match.group(4)),
                        'orientation': pin_match.group(5)
                    }
                    self.data['PINS'].append(pin)
                    logger.info(f"解析到引脚: {pin['name']}")
            
            # 解析网络
            net_pattern = r'NETS\s+(\d+)\s*;\s*(.*?)(?=END\s+NETS|$)'
            net_section = re.search(net_pattern, content, re.DOTALL)
            if net_section:
                net_content = net_section.group(2)
                net_item_pattern = r'NET\s+(\w+)\s+(\d+)\s+(\w+)\s+\(\s*(\w+)\s+(\w+)\s*\)'
                net_matches = re.finditer(net_item_pattern, net_content)
            
                for net_match in net_matches:
                    net = {
                        'name': net_match.group(1),
                        'num_pins': int(net_match.group(2)),
                        'type': net_match.group(3),
                        'source': net_match.group(4),
                        'target': net_match.group(5)
                    }
                    self.data['NETS'].append(net)
                    logger.info(f"解析到网络: {net['name']}")
            
            # 解析特殊网络
            specialnet_pattern = r'SPECIALNETS\s+(\d+)\s*;\s*(.*?)(?=END\s+SPECIALNETS|$)'
            specialnet_section = re.search(specialnet_pattern, content, re.DOTALL)
            if specialnet_section:
                specialnet_content = specialnet_section.group(2)
                specialnet_item_pattern = r'SPECIALNET\s+(\w+)\s+(\d+)\s+(\w+)\s+\(\s*(\w+)\s+(\w+)\s*\)'
                specialnet_matches = re.finditer(specialnet_item_pattern, specialnet_content)
            
                for specialnet_match in specialnet_matches:
                    specialnet = {
                        'name': specialnet_match.group(1),
                        'num_pins': int(specialnet_match.group(2)),
                        'type': specialnet_match.group(3),
                        'source': specialnet_match.group(4),
                        'target': specialnet_match.group(5)
                    }
                    self.data['SPECIALNETS'].append(specialnet)
                    logger.info(f"解析到特殊网络: {specialnet['name']}")
            
            logger.info("DEF文件解析完成")
            return self.data
                
        except Exception as e:
            logger.error(f"解析DEF文件时出错: {str(e)}")
            return None