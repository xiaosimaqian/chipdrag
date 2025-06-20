import re
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def split_by_token_limit(content: str, max_tokens: int = 1000) -> List[str]:
    """将LEF文件内容按token数量分割
    
    Args:
        content: LEF文件内容
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

def parse_large_lef_file(tech_lef_file: str, cells_lef_file: str = None, max_tokens: int = 1000) -> Dict[str, Any]:
    """解析大型LEF文件
    
    Args:
        tech_lef_file: 工艺LEF文件路径
        cells_lef_file: 单元库LEF文件路径（可选）
        max_tokens: 每个分片的最大token数量
        
    Returns:
        解析后的数据字典
    """
    logger.info(f"开始解析大型LEF文件: {tech_lef_file}")
    
    try:
        content = ""
        with open(tech_lef_file, 'r') as f:
            content += f.read() + "\n"
            
        if cells_lef_file:
            logger.info(f"开始解析单元库LEF文件: {cells_lef_file}")
            with open(cells_lef_file, 'r') as f:
                content += f.read() + "\n"
                
        # 分割文件内容，保持LEF语句的完整性
        statements = []
        current_statement = []
        current_tokens = 0
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 计算当前行的token数量
            line_tokens = len(line.split())
            
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
            
        logger.info(f"LEF文件已分割为 {len(statements)} 个语句")
        
        # 解析每个语句
        parser = LEFParser()
        result = {}
        
        for i, statement in enumerate(statements):
            logger.info(f"正在解析第 {i+1}/{len(statements)} 个语句")
            try:
                chunk_data = parser.parse_lef(statement)
                # 合并结果
                for key, value in chunk_data.items():
                    if key not in result:
                        result[key] = value
                    elif isinstance(value, dict):
                        result[key].update(value)
                    elif isinstance(value, list):
                        result[key].extend(value)
            except Exception as e:
                logger.error(f"解析第 {i+1} 个语句时出错: {str(e)}")
                continue
                
        logger.info("大型LEF文件解析完成")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"LEF文件不存在: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"解析LEF文件时出错: {str(e)}")
        raise

# 为向后兼容性提供别名
parse_lef_file = parse_large_lef_file

class LEFParser:
    def __init__(self):
        self.current_section = None
        self.data = {
            'VERSION': None,
            'DESIGN': None,
            'UNITS': {},
            'MACROS': {},
            'LAYERS': {},
            'VIAS': {},
            'SITES': {},
            'SPACING': {}
        }
        
    def parse_lef(self, content: str) -> Dict[str, Any]:
        """解析LEF文件内容
        
        Args:
            content: LEF文件内容
            
        Returns:
            解析后的数据字典
        """
        logger.info("开始解析LEF文件内容")
        
        try:
            # 解析版本和设计名称（仅在第一次解析时检查）
            if not self.data['VERSION']:
                version_match = re.search(r'VERSION\s+(\S+)', content)
                if version_match:
                    self.data['VERSION'] = version_match.group(1)
                    logger.info(f"解析到版本: {self.data['VERSION']}")
                    
            if not self.data['DESIGN']:
                design_match = re.search(r'DESIGN\s+(\S+)', content)
                if design_match:
                    self.data['DESIGN'] = design_match.group(1)
                    logger.info(f"解析到设计名称: {self.data['DESIGN']}")
            
            # 解析各个部分
            logger.info("开始解析UNITS部分")
            self._parse_units(content)
            
            logger.info("开始解析MACROS部分")
            self._parse_macros(content)
            
            logger.info("开始解析LAYERS部分")
            self._parse_layers(content)
            
            logger.info("开始解析VIAS部分")
            self._parse_vias(content)
            
            logger.info("开始解析SITES部分")
            self._parse_sites(content)
            
            logger.info("开始解析SPACING部分")
            self._parse_spacing(content)
            
            logger.info("开始解析MINFEATURE部分")
            self._parse_minfeature(content)
            
            logger.info("LEF文件解析完成")
            return self.data
            
        except Exception as e:
            logger.error(f"解析LEF文件时出错: {str(e)}")
            return None
            
    def _parse_units(self, content: str):
        """解析UNITS部分"""
        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if not units_match:
            logger.warning("LEF文件缺少UNITS声明")
            return
            
        try:
            self.data['UNITS'] = {
                'distance': int(units_match.group(1)),
                'unit': 'MICRONS'
            }
        except ValueError as e:
            logger.error(f"解析UNITS时出错: {str(e)}")
            raise
            
    def _parse_macros(self, content: str):
        """解析MACROS部分"""
        macro_sections = re.finditer(r'MACRO\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in macro_sections:
            try:
                macro_name = section.group(1)
                macro_content = section.group(0)
                
                # 解析大小
                size_match = re.search(r'SIZE\s+(\d+\.?\d*)\s+BY\s+(\d+\.?\d*)', macro_content)
                if not size_match:
                    logger.warning(f"宏单元 {macro_name} 缺少SIZE声明")
                    continue
                    
                size = {
                    'width': float(size_match.group(1)),
                    'height': float(size_match.group(2))
                }
                
                # 解析引脚
                pins = {}
                pin_sections = re.finditer(r'PIN\s+(\S+).*?END\s+\1', macro_content, re.DOTALL)
                
                for pin_section in pin_sections:
                    pin_name = pin_section.group(1)
                    pin_content = pin_section.group(0)
                    
                    # 解析引脚方向
                    direction_match = re.search(r'DIRECTION\s+(\S+)', pin_content)
                    direction = direction_match.group(1) if direction_match else 'INOUT'
                    
                    # 解析引脚用途
                    use_match = re.search(r'USE\s+(\S+)', pin_content)
                    use = use_match.group(1) if use_match else None
                    
                    # 解析引脚形状
                    shape_match = re.search(r'SHAPE\s+(\S+)', pin_content)
                    shape = shape_match.group(1) if shape_match else None
                    
                    # 解析引脚位置
                    port_sections = re.finditer(r'PORT.*?END', pin_content, re.DOTALL)
                    port_rects = []
                    
                    for port_section in port_sections:
                        rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', port_section.group(0))
                        for rect in rect_matches:
                            port_rects.append({
                                'x1': float(rect.group(1)),
                                'y1': float(rect.group(2)),
                                'x2': float(rect.group(3)),
                                'y2': float(rect.group(4))
                            })
                    
                    pins[pin_name] = {
                        'direction': direction,
                        'use': use,
                        'shape': shape,
                        'ports': port_rects
                    }
                
                # 解析障碍物
                obs_sections = re.finditer(r'OBS.*?END', macro_content, re.DOTALL)
                obs_rects = []
                
                for obs_section in obs_sections:
                    rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', obs_section.group(0))
                    for rect in rect_matches:
                        obs_rects.append({
                            'x1': float(rect.group(1)),
                            'y1': float(rect.group(2)),
                            'x2': float(rect.group(3)),
                            'y2': float(rect.group(4))
                        })
                
                # 解析对称性
                symmetry_match = re.search(r'SYMMETRY\s+(\S+)', macro_content)
                symmetry = symmetry_match.group(1) if symmetry_match else None
                
                # 解析类
                class_match = re.search(r'CLASS\s+(\S+)', macro_content)
                macro_class = class_match.group(1) if class_match else None
                
                self.data['MACROS'][macro_name] = {
                    'SIZE': size,
                    'PIN': pins,
                    'OBS': obs_rects,
                    'SYMMETRY': symmetry,
                    'CLASS': macro_class
                }
                
            except Exception as e:
                logger.error(f"解析宏单元 {macro_name} 时出错: {str(e)}")
                continue
                
    def _parse_layers(self, content: str):
        """解析LAYERS部分"""
        layer_sections = re.finditer(r'LAYER\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in layer_sections:
            try:
                layer_name = section.group(1)
                layer_content = section.group(0)
                
                # 解析层类型
                type_match = re.search(r'TYPE\s+(\S+)', layer_content)
                layer_type = type_match.group(1) if type_match else 'ROUTING'
                
                # 解析层属性
                properties = {}
                property_matches = re.finditer(r'(\S+)\s+(\S+)', layer_content)
                
                for match in property_matches:
                    prop_name = match.group(1)
                    prop_value = match.group(2)
                    properties[prop_name] = prop_value
                
                # 解析层方向
                direction_match = re.search(r'DIRECTION\s+(\S+)', layer_content)
                direction = direction_match.group(1) if direction_match else None
                
                # 解析层间距
                spacing_matches = re.finditer(r'SPACING\s+(\d+\.?\d*)', layer_content)
                spacing = [float(match.group(1)) for match in spacing_matches]
                
                # 解析层宽度
                width_match = re.search(r'WIDTH\s+(\d+\.?\d*)', layer_content)
                width = float(width_match.group(1)) if width_match else None
                
                self.data['LAYERS'][layer_name] = {
                    'TYPE': layer_type,
                    'DIRECTION': direction,
                    'WIDTH': width,
                    'SPACING': spacing,
                    'PROPERTIES': properties
                }
                
            except Exception as e:
                logger.error(f"解析层 {layer_name} 时出错: {str(e)}")
                continue
                
    def _parse_vias(self, content: str):
        """解析VIAS部分"""
        via_sections = re.finditer(r'VIA\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in via_sections:
            try:
                via_name = section.group(1)
                via_content = section.group(0)
                
                # 解析Via规则
                rule_match = re.search(r'VIARULE\s+(\S+)', via_content)
                rule = rule_match.group(1) if rule_match else None
                
                # 解析Via层
                layers = []
                layer_matches = re.finditer(r'LAYER\s+(\S+).*?END\s+\1', via_content, re.DOTALL)
                
                for layer_match in layer_matches:
                    layer_name = layer_match.group(1)
                    layer_content = layer_match.group(0)
                    
                    # 解析层矩形
                    rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', layer_content)
                    rects = []
                    
                    for rect in rect_matches:
                        rects.append({
                            'x1': float(rect.group(1)),
                            'y1': float(rect.group(2)),
                            'x2': float(rect.group(3)),
                            'y2': float(rect.group(4))
                        })
                    
                    layers.append({
                        'name': layer_name,
                        'rects': rects
                    })
                
                self.data['VIAS'][via_name] = {
                    'RULE': rule,
                    'LAYERS': layers
                }
                
            except Exception as e:
                logger.error(f"解析Via {via_name} 时出错: {str(e)}")
                continue
                
    def _parse_sites(self, content: str):
        """解析SITES部分"""
        site_sections = re.finditer(r'SITE\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in site_sections:
            try:
                site_name = section.group(1)
                site_content = section.group(0)
                
                # 解析站点类
                class_match = re.search(r'CLASS\s+(\S+)', site_content)
                site_class = class_match.group(1) if class_match else None
                
                # 解析站点大小
                size_match = re.search(r'SIZE\s+(\d+\.?\d*)\s+BY\s+(\d+\.?\d*)', site_content)
                if size_match:
                    size = {
                        'width': float(size_match.group(1)),
                        'height': float(size_match.group(2))
                    }
                else:
                    size = None
                
                # 解析站点对称性
                symmetry_match = re.search(r'SYMMETRY\s+(\S+)', site_content)
                symmetry = symmetry_match.group(1) if symmetry_match else None
                
                self.data['SITES'][site_name] = {
                    'CLASS': site_class,
                    'SIZE': size,
                    'SYMMETRY': symmetry
                }
                
            except Exception as e:
                logger.error(f"解析站点 {site_name} 时出错: {str(e)}")
                continue
                
    def _parse_spacing(self, content: str):
        """解析SPACING部分"""
        spacing_sections = re.finditer(r'SPACING\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in spacing_sections:
            try:
                spacing_name = section.group(1)
                spacing_content = section.group(0)
                
                # 解析间距规则
                rules = []
                rule_matches = re.finditer(r'SPACING\s+(\d+\.?\d*)', spacing_content)
                
                for rule_match in rule_matches:
                    rules.append(float(rule_match.group(1)))
                
                self.data['SPACING'][spacing_name] = {
                    'RULES': rules
                }
                
            except Exception as e:
                logger.error(f"解析间距规则 {spacing_name} 时出错: {str(e)}")
                continue
                
    def _parse_minfeature(self, content: str):
        """解析MINFEATURE部分"""
        minfeature_match = re.search(r'MINFEATURE\s+(\d+\.?\d*)\s+(\d+\.?\d*)', content)
        if minfeature_match:
            try:
                self.data['MINFEATURE'] = {
                    'width': float(minfeature_match.group(1)),
                    'length': float(minfeature_match.group(2))
                }
            except ValueError as e:
                logger.error(f"解析MINFEATURE时出错: {str(e)}")
                raise 