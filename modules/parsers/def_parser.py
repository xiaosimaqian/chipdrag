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

# 添加parse_def函数作为parse_large_def_file的别名
def parse_def(def_file: str) -> Dict[str, Any]:
    """解析DEF文件的便捷函数
    
    Args:
        def_file: DEF文件路径
        
    Returns:
        解析后的数据字典
    """
    return parse_large_def_file(def_file)

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
            
            for match in track_matches:
                self.data['TRACKS'].append({
                    'direction': match.group(1),
                    'start': int(match.group(2)),
                    'num_tracks': int(match.group(3)),
                    'step': int(match.group(4)),
                    'layer': match.group(5)
                })
            
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
            component_section_match = re.search(r'COMPONENTS\s+(\d+)\s*;', content)
            if not component_section_match:
                logger.info("在内容块中未找到COMPONENTS段。")
            else:
                num_components = int(component_section_match.group(1))
                logger.info(f"找到COMPONENTS段，声明了 {num_components} 个组件。")
                
                component_block_pattern = r'COMPONENTS\s+\d+\s*;\s*(.*?)\s*END\s+COMPONENTS'
                component_block_match = re.search(component_block_pattern, content, re.DOTALL)
                
                if not component_block_match:
                    logger.warning("找到了COMPONENTS段开始，但未找到'END COMPONENTS'。")
                else:
                    component_block = component_block_match.group(1)
                    # 解析每个组件
                    # 支持 PLACED, FIXED, UNPLACED
                    single_comp_pattern = r'-\s+([\w\d/\[\]]+)\s+([\w\d/]+)(?:\s*\+\s*(PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\S+))?(?:\s*\+\s*UNPLACED)?'
                    
                    comp_matches = re.finditer(single_comp_pattern, component_block)
                    
                    components_found = 0
                    for comp_match in comp_matches:
                        components_found += 1
                        comp_name = comp_match.group(1)
                        comp_model = comp_match.group(2)
                        status = comp_match.group(3)
                        
                        comp_data = {
                            'model': comp_model,
                            'name': comp_name,
                            'status': 'UNPLACED'
                        }

                        if status in ['PLACED', 'FIXED']:
                            comp_data.update({
                                'status': status,
                                'x': int(comp_match.group(4)),
                                'y': int(comp_match.group(5)),
                                'orientation': comp_match.group(6)
                            })
                        
                        self.data['COMPONENTS'].append(comp_data)

                    logger.info(f"从COMPONENTS段中成功解析了 {components_found} 个组件。")

            # 解析引脚
            pin_pattern = r'PINS\s+(\d+)\s*;'
            pin_section_match = re.search(pin_pattern, content, re.DOTALL)
            if pin_section_match:
                num_pins = int(pin_section_match.group(1))
                self.data['PINS'] = {}
                logger.info(f"开始解析PINS段，共 {num_pins} 个引脚")
            else:
                logger.warning("未找到PINS段。")
            
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

    def parse(self, content: str) -> Dict[str, Any]:
        """解析DEF文件内容

        Args:
            content: DEF文件内容字符串

        Returns:
            解析后的数据字典
        """
        self.data = {}
        self.current_section = None
        
        # 预先检查COMPONENTS段是否存在
        if 'COMPONENTS' not in content:
            logger.warning("DEF文件中未找到COMPONENTS段。")
        else:
            logger.info("DEF文件中存在COMPONENTS段。")
            
        lines = content.splitlines()
        
        in_components_section = False
        component_lines = 0

        for line in lines:
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('COMPONENTS'):
                self.current_section = 'COMPONENTS'
                num_components_str = line.split()[1]
                self.data['COMPONENTS'] = {}
                in_components_section = True
                logger.info(f"开始解析COMPONENTS段，声明数量: {num_components_str}")
            elif line.startswith('PINS'):
                if in_components_section:
                    logger.info(f"COMPONENTS段结束，共处理了 {component_lines} 行。")
                    in_components_section = False
                self.current_section = 'PINS'
                self.data['PINS'] = {}
                logger.info(f"开始解析PINS段，共 {line.split()[1]} 个引脚")
            elif self.current_section:
                self._parse_section_line(line)
                if self.current_section == 'COMPONENTS':
                    component_lines += 1
        
        if in_components_section:
            logger.info(f"COMPONENTS段在文件末尾结束，共处理了 {component_lines} 行。")

        logger.info(f"DEF文件解析完成。找到 {len(self.data.get('COMPONENTS', {}))} 个组件。")
        return self.data

    def _parse_version(self, line: str):
        self.data['VERSION'] = line.split()[1]
        logger.info(f"解析到版本: {self.data['VERSION']}")

    def _parse_design(self, line: str):
        self.data['DESIGN'] = line.split()[1]
        logger.info(f"解析到设计名称: {self.data['DESIGN']}")

    def _parse_units(self, line: str):
        match = re.search(r'UNITS DISTANCE MICRONS (\d+)', line)
        if match:
            self.data['UNITS'] = {'distance': int(match.group(1)), 'unit': 'MICRONS'}
            logger.info(f"解析到单位: {self.data['UNITS']}")

    def _parse_diearea(self, line: str):
        coords_str = re.findall(r'\( (\d+) (\d+) \)', line)
        if len(coords_str) == 2:
            x1, y1 = map(int, coords_str[0])
            x2, y2 = map(int, coords_str[1])
            self.data['DIEAREA'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            logger.info("解析到芯片区域")

    def _parse_row(self, line: str):
        parts = line.split()
        row_name = parts[1]
        self.data.setdefault('ROWS', {})[row_name] = {
            'site': parts[2],
            'x': int(parts[4]),
            'y': int(parts[5]),
            'orient': parts[7],
            'do': int(parts[9]),
            'by': int(parts[11]),
            'step_x': int(parts[13]),
            'step_y': int(parts[15]),
        }
        logger.debug(f"解析到行: {row_name}")

    def _parse_tracks(self, line: str):
        parts = line.split()
        direction = parts[1]
        start = int(parts[2])
        num_tracks = int(parts[4])
        pitch = int(parts[6])
        layer = parts[8]
        self.data.setdefault('TRACKS', []).append({
            'direction': direction,
            'start': start,
            'num_tracks': num_tracks,
            'pitch': pitch,
            'layer': layer,
        })
        logger.info(f"解析到轨道: {direction} {layer}")

    def _parse_gcellgrid(self, line: str):
        parts = line.split()
        direction = parts[1]
        start = int(parts[3])
        num_grids = int(parts[5])
        step = int(parts[7])
        self.data.setdefault('GCELLGRID', []).append({
            'direction': direction,
            'start': start,
            'num_grids': num_grids,
            'step': step,
        })
        logger.info(f"解析到网格: {direction}")

    def _parse_section_line(self, line: str):
        # Implementation of _parse_section_line method
        pass