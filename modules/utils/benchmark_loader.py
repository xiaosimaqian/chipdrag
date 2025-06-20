import os
import json
import re
import logging
import time
from typing import Dict, Any, Optional
from ..parsers.def_parser import DEFParser
from ..parsers.lef_parser import LEFParser
from ..parsers.verilog_parser import VerilogParser

logger = logging.getLogger(__name__)

class BenchmarkLoader:
    def __init__(self, benchmark_path: str):
        """初始化基准测试加载器
        
        Args:
            benchmark_path: 基准测试数据目录路径
        """
        if not os.path.exists(benchmark_path):
            raise ValueError(f"基准测试数据目录不存在: {benchmark_path}")
            
        self.benchmark_dir = benchmark_path
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_design(self, design_name: str) -> Dict:
        """加载设计信息
        
        Args:
            design_name: 设计名称
            
        Returns:
            设计信息字典
            
        Raises:
            ValueError: 当设计目录不存在时
            FileNotFoundError: 当必需的文件不存在时
        """
        logger.info(f"开始加载设计: {design_name}")
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f'{design_name}_cache.json')
        if os.path.exists(cache_file):
            logger.info(f"发现缓存文件，正在加载: {cache_file}")
            start_time = time.time()
            try:
                with open(cache_file, 'r') as f:
                    design_info = json.load(f)
                logger.info(f"从缓存加载完成，耗时: {time.time() - start_time:.2f}秒")
                return design_info
            except Exception as e:
                logger.warning(f"缓存文件加载失败: {e}")
        
        # 检查设计目录
        design_dir = os.path.join(self.benchmark_dir, design_name)
        if not os.path.exists(design_dir):
            raise ValueError(f"设计目录不存在: {design_dir}")
            
        # 检查必需的文件
        required_files = {
            'verilog': os.path.join(design_dir, 'design.v'),
            'def': os.path.join(design_dir, 'floorplan.def'),
            'cells_lef': os.path.join(design_dir, 'cells.lef'),
            'tech_lef': os.path.join(design_dir, 'tech.lef'),
            'constraints': os.path.join(design_dir, 'placement.constraints')
        }
        
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"必需的文件不存在: {file_path}")
        
        try:
            # 加载各种文件
            verilog_info = self._load_verilog(required_files['verilog'])
            def_info = self._load_def(required_files['def'])
            cells_lef_info = self._load_lef(required_files['cells_lef'])
            tech_lef_info = self._load_lef(required_files['tech_lef'])
            constraints_info = self._load_constraints(required_files['constraints'])
            
            # 合并设计信息
            design_info = {
                'name': design_name,
                'netlist': verilog_info,
                'def': def_info,
                'lef': {
                    'cells': cells_lef_info,
                    'technology': tech_lef_info
                },
                'constraints': constraints_info
            }
            
            # 保存缓存
            logger.info(f"正在保存中间结果到: {cache_file}")
            start_time = time.time()
            with open(cache_file, 'w') as f:
                json.dump(design_info, f, indent=2)
            logger.info(f"中间结果保存完成，耗时: {time.time() - start_time:.2f}秒")
            
            return design_info
            
        except Exception as e:
            logger.error(f"加载设计失败: {e}")
            raise
        
    def _load_verilog(self, verilog_file: str) -> Dict:
        """加载Verilog网表文件
        
        Args:
            verilog_file: Verilog文件路径
            
        Returns:
            网表信息字典
        """
        logger.info(f"正在读取Verilog文件: {verilog_file}")
        start_time = time.time()
        
        with open(verilog_file, 'r') as f:
            content = f.read()
            
        # 提取模块定义
        modules = []
        module_pattern = r'module\s+(\w+)\s*\((.*?)\);.*?endmodule'
        for match in re.finditer(module_pattern, content, re.DOTALL):
            module_name = match.group(1)
            ports = self._parse_ports(match.group(2))
            instances = self._parse_instances(match.group(0))
            nets = self._parse_nets(match.group(0))
            
            modules.append({
                'name': module_name,
                'ports': ports,
                'instances': instances,
                'nets': nets
            })
            
        verilog_info = {
            'modules': modules,
            'attributes': {
                'file': verilog_file,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"Verilog文件读取完成，耗时: {time.time() - start_time:.2f}秒")
        return verilog_info
        
    def _load_def(self, def_file: str) -> Dict:
        """加载DEF文件
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            DEF信息字典
        """
        logger.info(f"正在读取DEF文件: {def_file}")
        start_time = time.time()
        
        with open(def_file, 'r') as f:
            content = f.read()
            
        # 提取单位信息
        units = self._parse_def_units(content)
        
        # 提取组件信息
        components = self._parse_def_components(content)
        
        # 提取区域信息
        die_area = self._parse_def_die_area(content)
        
        # 提取行信息
        rows = self._parse_def_rows(content)
        
        # 提取轨道信息
        tracks = self._parse_def_tracks(content)
        
        def_info = {
            'units': units,
            'components': components,
            'die_area': die_area,
            'rows': rows,
            'tracks': tracks,
            'attributes': {
                'file': def_file,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"DEF文件读取完成，耗时: {time.time() - start_time:.2f}秒")
        return def_info
        
    def _load_lef(self, lef_file: str) -> Dict:
        """加载LEF文件
        
        Args:
            lef_file: LEF文件路径
            
        Returns:
            LEF信息字典
        """
        logger.info(f"正在读取LEF文件: {lef_file}")
        start_time = time.time()
        
        with open(lef_file, 'r') as f:
            content = f.read()
            
        # 判断是单元库还是工艺文件
        is_tech = 'TECHNOLOGY' in content
        
        if is_tech:
            # 提取工艺信息
            tech_info = self._parse_tech_lef(content)
            lef_info = {
                'technology': tech_info,
                'attributes': {
                    'file': lef_file,
                    'type': 'technology',
                    'timestamp': time.time()
                }
            }
        else:
            # 提取单元信息
            cells = self._parse_cell_lef(content)
            lef_info = {
                'cells': cells,
                'attributes': {
                    'file': lef_file,
                    'type': 'cell',
                    'timestamp': time.time()
                }
            }
            
        logger.info(f"LEF文件读取完成，耗时: {time.time() - start_time:.2f}秒")
        return lef_info
        
    def _load_constraints(self, constraints_file: str) -> Dict:
        """加载约束文件
        
        Args:
            constraints_file: 约束文件路径
            
        Returns:
            约束信息字典
        """
        logger.info(f"正在读取约束文件: {constraints_file}")
        start_time = time.time()
        
        with open(constraints_file, 'r') as f:
            content = f.read()
            
        # 解析约束
        constraints = self._parse_constraints(content)
        
        constraints_info = {
            'placement': constraints.get('placement', []),
            'timing': constraints.get('timing', []),
            'power': constraints.get('power', []),
            'attributes': {
                'file': constraints_file,
                'timestamp': time.time()
            }
        }
        
        logger.info(f"约束文件读取完成，耗时: {time.time() - start_time:.2f}秒")
        return constraints_info
        
    def _parse_ports(self, ports_str: str) -> Dict:
        """解析端口定义
        
        Args:
            ports_str: 端口定义字符串
            
        Returns:
            端口列表
        """
        ports = {}
        port_pattern = r'(input|output|inout)\s+(\w+)'
        for match in re.finditer(port_pattern, ports_str):
            direction = match.group(1)
            name = match.group(2)
            ports[name] = direction
        return ports
        
    def _parse_instances(self, module_str: str) -> Dict:
        """解析实例定义
        
        Args:
            module_str: 模块定义字符串
            
        Returns:
            实例列表
        """
        instances = {}
        instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\);'
        for match in re.finditer(instance_pattern, module_str):
            cell_type = match.group(1)
            instance_name = match.group(2)
            pins = self._parse_pins(match.group(3))
            instances[instance_name] = {
                'type': cell_type,
                'pins': pins
            }
        return instances
        
    def _parse_nets(self, module_str: str) -> Dict:
        """解析网络定义
        
        Args:
            module_str: 模块定义字符串
            
        Returns:
            网络列表
        """
        nets = {}
        net_pattern = r'wire\s+(\w+);'
        for match in re.finditer(net_pattern, module_str):
            net_name = match.group(1)
            nets[net_name] = []
        return nets
        
    def _parse_pins(self, pins_str: str) -> Dict:
        """解析引脚连接
        
        Args:
            pins_str: 引脚连接字符串
            
        Returns:
            引脚列表
        """
        pins = {}
        pin_pattern = r'\.(\w+)\s*\(\s*(\w+)\s*\)'
        for match in re.finditer(pin_pattern, pins_str):
            pin_name = match.group(1)
            net_name = match.group(2)
            pins[pin_name] = net_name
        return pins
        
    def _parse_def_units(self, def_content: str) -> Dict:
        """解析DEF单位信息
        
        Args:
            def_content: DEF文件内容
            
        Returns:
            单位信息字典
        """
        units = {}
        unit_pattern = r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)'
        match = re.search(unit_pattern, def_content)
        if match:
            units['distance'] = int(match.group(1))
        return units
        
    def _parse_def_components(self, def_content: str) -> Dict:
        """解析DEF组件信息
        
        Args:
            def_content: DEF文件内容
            
        Returns:
            组件列表
        """
        components = []
        comp_pattern = r'-\s+(\w+)\s+(\w+)\s+\+\s+PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)\s+(\w+)'
        for match in re.finditer(comp_pattern, def_content):
            name = match.group(1)
            cell_type = match.group(2)
            x = int(match.group(3))
            y = int(match.group(4))
            orientation = match.group(5)
            components.append({
                'name': name,
                'type': cell_type,
                'x': x,
                'y': y,
                'orientation': orientation
            })
        return components
        
    def _parse_def_die_area(self, def_content: str) -> Dict:
        """解析DEF区域信息
        
        Args:
            def_content: DEF文件内容
            
        Returns:
            区域信息字典
        """
        die_area = {}
        area_pattern = r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)'
        match = re.search(area_pattern, def_content)
        if match:
            die_area = {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
                'width': int(match.group(3)) - int(match.group(1)),
                'height': int(match.group(4)) - int(match.group(2))
            }
        return die_area
        
    def _parse_def_rows(self, def_content: str) -> Dict:
        """解析DEF行信息
        
        Args:
            def_content: DEF文件内容
            
        Returns:
            行列表
        """
        rows = []
        row_pattern = r'ROW\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\w+)\s+DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+)'
        for match in re.finditer(row_pattern, def_content):
            rows.append({
                'name': match.group(1),
                'site': match.group(2),
                'x': int(match.group(3)),
                'y': int(match.group(4)),
                'orientation': match.group(5),
                'num_x': int(match.group(6)),
                'num_y': int(match.group(7)),
                'step_x': int(match.group(8)),
                'step_y': int(match.group(9))
            })
        return rows
        
    def _parse_def_tracks(self, def_content: str) -> Dict:
        """解析DEF轨道信息
        
        Args:
            def_content: DEF文件内容
            
        Returns:
            轨道列表
        """
        tracks = []
        track_pattern = r'TRACKS\s+(\w+)\s+(\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\w+)'
        for match in re.finditer(track_pattern, def_content):
            tracks.append({
                'direction': match.group(1),
                'start': int(match.group(2)),
                'num_tracks': int(match.group(3)),
                'step': int(match.group(4)),
                'layer': match.group(5)
            })
        return tracks
        
    def _parse_tech_lef(self, lef_content: str) -> Dict:
        """解析工艺LEF文件
        
        Args:
            lef_content: LEF文件内容
            
        Returns:
            工艺信息字典
        """
        tech_info = {}
        
        # 解析层信息
        layers = []
        layer_pattern = r'LAYER\s+(\w+)\s+TYPE\s+(\w+)'
        for match in re.finditer(layer_pattern, lef_content):
            layers.append({
                'name': match.group(1),
                'type': match.group(2)
            })
        tech_info['layers'] = layers
        
        # 解析间距规则
        spacing_rules = []
        spacing_pattern = r'SPACING\s+(\d+)'
        for match in re.finditer(spacing_pattern, lef_content):
            spacing_rules.append(int(match.group(1)))
        tech_info['spacing_rules'] = spacing_rules
        
        return tech_info
        
    def _parse_cell_lef(self, lef_content: str) -> Dict:
        """解析单元库LEF文件
        
        Args:
            lef_content: LEF文件内容
            
        Returns:
            单元列表
        """
        cells = []
        cell_pattern = r'MACRO\s+(\w+).*?END\s+\1'
        for match in re.finditer(cell_pattern, lef_content, re.DOTALL):
            cell_content = match.group(0)
            cell_name = match.group(1)
            
            # 解析单元大小
            size_pattern = r'SIZE\s+(\d+)\s+BY\s+(\d+)'
            size_match = re.search(size_pattern, cell_content)
            size = {
                'width': int(size_match.group(1)) if size_match else 0,
                'height': int(size_match.group(2)) if size_match else 0
            }
            
            # 解析引脚
            pins = []
            pin_pattern = r'PIN\s+(\w+).*?END\s+\1'
            for pin_match in re.finditer(pin_pattern, cell_content, re.DOTALL):
                pin_name = pin_match.group(1)
                pins.append({
                    'name': pin_name,
                    'direction': self._parse_pin_direction(pin_match.group(0))
                })
                
            cells.append({
                'name': cell_name,
                'size': size,
                'pins': pins
            })
            
        return cells
        
    def _parse_pin_direction(self, pin_content: str) -> str:
        """解析引脚方向
        
        Args:
            pin_content: 引脚定义内容
            
        Returns:
            引脚方向
        """
        direction_pattern = r'DIRECTION\s+(\w+)'
        match = re.search(direction_pattern, pin_content)
        return match.group(1) if match else 'UNKNOWN'
        
    def _parse_constraints(self, constraints_content: str) -> Dict:
        """解析约束文件
        
        Args:
            constraints_content: 约束文件内容
            
        Returns:
            约束信息字典
        """
        constraints = {
            'placement': [],
            'timing': [],
            'power': []
        }
        
        # 解析布局约束
        placement_pattern = r'PLACEMENT\s+(\w+)\s+(\d+)\s+(\d+)'
        for match in re.finditer(placement_pattern, constraints_content):
            constraints['placement'].append({
                'instance': match.group(1),
                'x': int(match.group(2)),
                'y': int(match.group(3))
            })
            
        # 解析时序约束
        timing_pattern = r'TIMING\s+(\w+)\s+(\d+)'
        for match in re.finditer(timing_pattern, constraints_content):
            constraints['timing'].append({
                'path': match.group(1),
                'delay': int(match.group(2))
            })
            
        # 解析功耗约束
        power_pattern = r'POWER\s+(\w+)\s+(\d+)'
        for match in re.finditer(power_pattern, constraints_content):
            constraints['power'].append({
                'instance': match.group(1),
                'max_power': int(match.group(2))
            })
            
        return constraints

    def load_benchmark(self, case: str) -> Dict:
        """加载基准测试用例
        
        Args:
            case: 基准测试用例名称
            
        Returns:
            基准测试用例信息字典
            
        Raises:
            ValueError: 当基准测试用例不存在时
        """
        try:
            return self.load_design(case)
        except Exception as e:
            logger.error(f"加载基准测试用例失败: {e}")
            raise ValueError(f"基准测试用例不存在或加载失败: {case}") 