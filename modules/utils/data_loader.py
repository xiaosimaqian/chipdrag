import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from modules.parsers.def_parser import DEFParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChipDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """初始化数据集
        
        Args:
            data_dir: 数据目录
            transform: 数据转换函数
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """加载样本
        
        Returns:
            List[Dict]: 样本列表
        """
        samples = []
        for case_dir in self.data_dir.iterdir():
            if case_dir.is_dir():
                sample = self._load_case(case_dir)
                if sample is not None:
                    samples.append(sample)
        return samples
    
    def _load_case(self, case_dir: Path) -> Optional[Dict]:
        """加载单个测试用例
        
        Args:
            case_dir: 用例目录
            
        Returns:
            Optional[Dict]: 用例数据
        """
        try:
            # 加载网表
            netlist_file = case_dir / "netlist.json"
            if not netlist_file.exists():
                return None
            with open(netlist_file, 'r') as f:
                netlist = json.load(f)
                
            # 加载布局定义
            def_file = case_dir / "layout.def"
            if not def_file.exists():
                return None
            def_data = self._load_def_file(def_file)
            
            # 加载约束
            constraints_file = case_dir / "constraints.json"
            if not constraints_file.exists():
                return None
            with open(constraints_file, 'r') as f:
                constraints = json.load(f)
                
            return {
                'name': case_dir.name,
                'netlist': netlist,
                'def_file': def_data,
                'constraints': constraints
            }
        except Exception as e:
            print(f"Error loading case {case_dir.name}: {str(e)}")
            return None
    
    def _load_def_file(self, def_file: Path) -> Dict:
        """加载DEF文件
        
        Args:
            def_file: DEF文件路径
            
        Returns:
            Dict: DEF文件数据
        """
        def_data = {
            'layout': {
                'die_area': None,
                'rows': [],
                'components': [],
                'nets': []
            }
        }
        
        try:
            # 使用 DEFParser 解析文件
            parser = DEFParser()
            def_data = parser.parse_def(def_file)
            
            # 转换为所需的格式
            layout_data = {
                'layout': {
                    'die_area': {
                        'width': def_data['DIEAREA']['width'],
                        'height': def_data['DIEAREA']['height']
                    },
                    'rows': [],
                    'components': [],
                    'nets': []
                }
            }
            
            # 转换行信息
            if 'ROWS' in def_data:
                for row in def_data['ROWS']:
                    layout_data['layout']['rows'].append({
                        'name': row['name'],
                        'site': row['site'],
                        'x': row['x'],
                        'y': row['y'],
                        'orientation': row['orientation'],
                        'num_x': row['numx'],
                        'num_y': row['numy']
                    })
            
            # 转换组件信息
            if 'COMPONENTS' in def_data:
                for comp_id, comp in def_data['COMPONENTS']['instances'].items():
                    if comp['placement']:
                        layout_data['layout']['components'].append({
                            'name': comp_id,
                            'type': comp['type'],
                            'x': comp['placement']['x'],
                            'y': comp['placement']['y'],
                            'orientation': comp['placement']['orientation']
                        })
            
            # 转换网络信息
            if 'NETS' in def_data:
                for net in def_data['NETS']['nets']:
                    layout_data['layout']['nets'].append({
                        'name': net['id'],
                        'connections': net['connections']
                    })
            
            return layout_data
            
        except Exception as e:
            logger.error(f"加载DEF文件失败: {str(e)}")
            raise
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class DataLoader:
    def __init__(self, config: Dict):
        """初始化数据加载器
        
        Args:
            config: 配置信息
        """
        self.config = config
        
    def load_case(self, test_case: Dict) -> Dict:
        """加载测试用例
        
        Args:
            test_case: 用例信息
            
        Returns:
            Dict: 用例数据
        """
        case_path = Path(test_case['path'])
        
        # 加载网表
        with open(case_path / test_case['netlist'], 'r') as f:
            netlist = json.load(f)
            
        # 加载布局定义
        def_file = case_path / test_case['layout']
        def_data = self._load_def_file(def_file)
        
        # 加载约束
        with open(case_path / test_case['constraints'], 'r') as f:
            constraints = json.load(f)
            
        return {
            'name': test_case['name'],
            'netlist': netlist,
            'def_file': def_data,
            'constraints': constraints
        }
    
    def _load_def_file(self, file_path: str) -> Dict:
        """加载DEF文件
        
        Args:
            file_path: DEF文件路径
            
        Returns:
            Dict: DEF文件数据
        """
        try:
            # 使用 DEFParser 解析文件
            parser = DEFParser()
            def_data = parser.parse_def(file_path)
            
            # 转换为所需的格式
            layout_data = {
                'layout': {
                    'die_area': {
                        'width': def_data['DIEAREA']['width'],
                        'height': def_data['DIEAREA']['height']
                    },
                    'rows': [],
                    'components': [],
                    'nets': []
                }
            }
            
            # 转换行信息
            if 'ROWS' in def_data:
                for row in def_data['ROWS']:
                    layout_data['layout']['rows'].append({
                        'name': row['name'],
                        'site': row['site'],
                        'x': row['x'],
                        'y': row['y'],
                        'orientation': row['orientation'],
                        'num_x': row['numx'],
                        'num_y': row['numy']
                    })
            
            # 转换组件信息
            if 'COMPONENTS' in def_data:
                for comp_id, comp in def_data['COMPONENTS']['instances'].items():
                    if comp['placement']:
                        layout_data['layout']['components'].append({
                            'name': comp_id,
                            'type': comp['type'],
                            'x': comp['placement']['x'],
                            'y': comp['placement']['y'],
                            'orientation': comp['placement']['orientation']
                        })
            
            # 转换网络信息
            if 'NETS' in def_data:
                for net in def_data['NETS']['nets']:
                    layout_data['layout']['nets'].append({
                        'name': net['id'],
                        'connections': net['connections']
                    })
            
            return layout_data
            
        except Exception as e:
            logger.error(f"加载DEF文件失败: {str(e)}")
            raise 