#!/usr/bin/env python3
"""
简化真实数据收集器
演示如何收集芯片设计领域的真实数据
"""

import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import re

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataCollector:
    """简化数据收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/real'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"简化数据收集器初始化完成，数据目录: {self.data_dir}")
    
    def collect_from_github(self, search_terms: List[str], max_repos: int = 10) -> List[Dict[str, Any]]:
        """从GitHub收集开源设计项目数据
        
        Args:
            search_terms: 搜索关键词
            max_repos: 最大仓库数量
            
        Returns:
            List[Dict[str, Any]]: 收集的数据
        """
        logger.info(f"开始从GitHub收集数据，搜索关键词: {search_terms}")
        
        collected_data = []
        
        for term in search_terms:
            try:
                # 搜索仓库
                repos = self._search_github_repos(term, max_repos)
                
                for repo in repos:
                    # 获取仓库信息
                    repo_data = self._get_repo_info(repo)
                    if repo_data:
                        collected_data.append(repo_data)
                    
                    # 获取设计文件
                    design_files = self._get_design_files(repo)
                    collected_data.extend(design_files)
                    
                    # 避免API限制
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"从GitHub收集数据失败，关键词 {term}: {str(e)}")
        
        logger.info(f"从GitHub收集到 {len(collected_data)} 条数据")
        return collected_data
    
    def _search_github_repos(self, search_term: str, max_repos: int) -> List[str]:
        """搜索GitHub仓库
        
        Args:
            search_term: 搜索关键词
            max_repos: 最大仓库数量
            
        Returns:
            List[str]: 仓库列表
        """
        repos = []
        
        try:
            # 构建搜索URL
            search_url = f"https://api.github.com/search/repositories"
            params = {
                'q': f'{search_term} language:verilog language:vhdl',
                'sort': 'stars',
                'order': 'desc',
                'per_page': min(max_repos, 30)
            }
            
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                repos = [item['full_name'] for item in data.get('items', [])]
                logger.info(f"找到 {len(repos)} 个相关仓库")
            else:
                logger.warning(f"GitHub API请求失败: {response.status_code}")
        
        except Exception as e:
            logger.error(f"搜索GitHub仓库失败: {str(e)}")
        
        return repos
    
    def _get_repo_info(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """获取仓库信息
        
        Args:
            repo_name: 仓库名称
            
        Returns:
            Optional[Dict[str, Any]]: 仓库信息
        """
        try:
            url = f"https://api.github.com/repos/{repo_name}"
            response = requests.get(url)
            
            if response.status_code == 200:
                repo_data = response.json()
                
                return {
                    'type': 'github_repo',
                    'name': repo_name,
                    'description': repo_data.get('description', ''),
                    'language': repo_data.get('language', ''),
                    'stars': repo_data.get('stargazers_count', 0),
                    'forks': repo_data.get('forks_count', 0),
                    'created_at': repo_data.get('created_at', ''),
                    'updated_at': repo_data.get('updated_at', ''),
                    'topics': repo_data.get('topics', []),
                    'url': repo_data.get('html_url', ''),
                    'timestamp': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"获取仓库信息失败 {repo_name}: {str(e)}")
        
        return None
    
    def _get_design_files(self, repo_name: str) -> List[Dict[str, Any]]:
        """获取设计文件信息
        
        Args:
            repo_name: 仓库名称
            
        Returns:
            List[Dict[str, Any]]: 设计文件列表
        """
        files = []
        
        try:
            # 获取仓库内容
            url = f"https://api.github.com/repos/{repo_name}/contents"
            response = requests.get(url)
            
            if response.status_code == 200:
                contents = response.json()
                
                for item in contents:
                    if item['type'] == 'file':
                        if self._is_design_file(item['name']):
                            file_info = {
                                'type': 'design_file',
                                'repo': repo_name,
                                'filename': item['name'],
                                'path': item['path'],
                                'size': item['size'],
                                'download_url': item['download_url'],
                                'file_type': self._get_file_type(item['name']),
                                'timestamp': datetime.now().isoformat()
                            }
                            files.append(file_info)
        
        except Exception as e:
            logger.error(f"获取设计文件失败 {repo_name}: {str(e)}")
        
        return files
    
    def _is_design_file(self, filename: str) -> bool:
        """判断是否为设计文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否为设计文件
        """
        design_extensions = ['.v', '.vhd', '.def', '.lef', '.lib', '.sdc', '.tcl', '.gds']
        return any(filename.lower().endswith(ext) for ext in design_extensions)
    
    def _get_file_type(self, filename: str) -> str:
        """获取文件类型
        
        Args:
            filename: 文件名
            
        Returns:
            str: 文件类型
        """
        ext = Path(filename).suffix.lower()
        
        file_types = {
            '.v': 'verilog',
            '.vhd': 'vhdl',
            '.def': 'def',
            '.lef': 'lef',
            '.lib': 'lib',
            '.sdc': 'sdc',
            '.tcl': 'tcl',
            '.gds': 'gds'
        }
        
        return file_types.get(ext, 'unknown')
    
    def collect_from_local_files(self, local_dir: str) -> List[Dict[str, Any]]:
        """从本地文件收集数据
        
        Args:
            local_dir: 本地目录路径
            
        Returns:
            List[Dict[str, Any]]: 收集的数据
        """
        logger.info(f"开始从本地目录收集数据: {local_dir}")
        
        collected_data = []
        local_path = Path(local_dir)
        
        if not local_path.exists():
            logger.error(f"本地目录不存在: {local_dir}")
            return collected_data
        
        # 查找设计文件
        design_extensions = ['.v', '.vhd', '.def', '.lef', '.lib', '.sdc', '.tcl']
        
        for ext in design_extensions:
            for file_path in local_path.rglob(f'*{ext}'):
                try:
                    file_data = self._parse_local_file(file_path)
                    if file_data:
                        collected_data.append(file_data)
                except Exception as e:
                    logger.error(f"解析本地文件失败 {file_path}: {str(e)}")
        
        logger.info(f"从本地目录收集到 {len(collected_data)} 条数据")
        return collected_data
    
    def _parse_local_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析本地文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[Dict[str, Any]]: 文件信息
        """
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.v':
                return self._parse_verilog_file(file_path)
            elif file_ext == '.vhd':
                return self._parse_vhdl_file(file_path)
            elif file_ext == '.def':
                return self._parse_def_file(file_path)
            else:
                return self._parse_generic_file(file_path)
                
        except Exception as e:
            logger.error(f"解析本地文件失败 {file_path}: {str(e)}")
            return None
    
    def _parse_verilog_file(self, file_path: Path) -> Dict[str, Any]:
        """解析Verilog文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取模块信息
            module_name = self._extract_module_name(content)
            port_count = self._extract_port_count(content)
            instance_count = self._extract_instance_count(content)
            
            return {
                'type': 'verilog_file',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'module_name': module_name,
                'port_count': port_count,
                'instance_count': instance_count,
                'line_count': len(content.splitlines()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析Verilog文件失败 {file_path}: {str(e)}")
            return {}
    
    def _parse_vhdl_file(self, file_path: Path) -> Dict[str, Any]:
        """解析VHDL文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取实体信息
            entity_name = self._extract_entity_name(content)
            port_count = self._extract_vhdl_port_count(content)
            
            return {
                'type': 'vhdl_file',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'entity_name': entity_name,
                'port_count': port_count,
                'line_count': len(content.splitlines()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析VHDL文件失败 {file_path}: {str(e)}")
            return {}
    
    def _parse_def_file(self, file_path: Path) -> Dict[str, Any]:
        """解析DEF文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取布局信息
            design_name = self._extract_def_design_name(content)
            component_count = self._extract_component_count(content)
            net_count = self._extract_net_count(content)
            
            return {
                'type': 'def_file',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'design_name': design_name,
                'component_count': component_count,
                'net_count': net_count,
                'line_count': len(content.splitlines()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析DEF文件失败 {file_path}: {str(e)}")
            return {}
    
    def _parse_generic_file(self, file_path: Path) -> Dict[str, Any]:
        """解析通用文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict[str, Any]: 文件信息
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'type': 'generic_file',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_extension': file_path.suffix,
                'line_count': len(content.splitlines()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析通用文件失败 {file_path}: {str(e)}")
            return {}
    
    def _extract_module_name(self, content: str) -> str:
        """提取模块名"""
        match = re.search(r'module\s+(\w+)', content)
        return match.group(1) if match else 'unknown'
    
    def _extract_port_count(self, content: str) -> int:
        """提取端口数量"""
        ports = re.findall(r'(input|output|inout)\s+', content)
        return len(ports)
    
    def _extract_instance_count(self, content: str) -> int:
        """提取实例数量"""
        instances = re.findall(r'(\w+)\s+(\w+)\s*\(', content)
        return len(instances)
    
    def _extract_entity_name(self, content: str) -> str:
        """提取实体名"""
        match = re.search(r'entity\s+(\w+)', content, re.IGNORECASE)
        return match.group(1) if match else 'unknown'
    
    def _extract_vhdl_port_count(self, content: str) -> int:
        """提取VHDL端口数量"""
        ports = re.findall(r'(\w+)\s*:\s*(in|out|inout)', content, re.IGNORECASE)
        return len(ports)
    
    def _extract_def_design_name(self, content: str) -> str:
        """提取DEF设计名"""
        match = re.search(r'DESIGN\s+(\w+)', content)
        return match.group(1) if match else 'unknown'
    
    def _extract_component_count(self, content: str) -> int:
        """提取组件数量"""
        components = re.findall(r'COMPONENTS\s+(\d+)', content)
        return int(components[0]) if components else 0
    
    def _extract_net_count(self, content: str) -> int:
        """提取网络数量"""
        nets = re.findall(r'NETS\s+(\d+)', content)
        return int(nets[0]) if nets else 0
    
    def generate_sample_queries(self, num_queries: int = 100) -> List[Dict[str, Any]]:
        """生成示例查询数据
        
        Args:
            num_queries: 查询数量
            
        Returns:
            List[Dict[str, Any]]: 查询列表
        """
        logger.info(f"生成 {num_queries} 个示例查询")
        
        queries = []
        design_types = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
        constraint_types = ['timing', 'power', 'area', 'reliability', 'yield']
        
        query_templates = [
            "Generate layout for {design_type} with {constraints} constraints",
            "Design {design_type} layout optimized for {constraints}",
            "Create {design_type} layout with focus on {constraints}",
            "Optimize {design_type} layout for {constraints} requirements"
        ]
        
        import random
        
        for i in range(num_queries):
            design_type = random.choice(design_types)
            num_constraints = random.randint(1, 3)
            constraints = random.sample(constraint_types, num_constraints)
            
            template = random.choice(query_templates)
            query_text = template.format(
                design_type=design_type,
                constraints=', '.join(constraints)
            )
            
            query = {
                'query_id': f'query_{i:04d}',
                'query_text': query_text,
                'design_type': design_type,
                'constraints': constraints,
                'complexity': random.uniform(0.3, 0.9),
                'user_id': f'user_{random.randint(1, 10):02d}',
                'timestamp': datetime.now().isoformat(),
                'source': 'synthetic'
            }
            
            queries.append(query)
        
        return queries
    
    def save_collected_data(self, data: List[Dict[str, Any]], data_type: str):
        """保存收集的数据
        
        Args:
            data: 收集的数据
            data_type: 数据类型
        """
        if not data:
            logger.warning("没有数据需要保存")
            return
        
        # 保存到JSON文件
        output_file = self.data_dir / f'{data_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到 {output_file}")
        
        # 生成统计报告
        self._generate_statistics(data, data_type)
    
    def _generate_statistics(self, data: List[Dict[str, Any]], data_type: str):
        """生成数据统计
        
        Args:
            data: 数据列表
            data_type: 数据类型
        """
        if not data:
            return
        
        # 统计信息
        stats = {
            'data_type': data_type,
            'total_count': len(data),
            'collection_timestamp': datetime.now().isoformat()
        }
        
        # 根据数据类型生成特定统计
        if data_type == 'github_repos':
            stats['language_distribution'] = {}
            stats['topic_distribution'] = {}
            
            for item in data:
                if 'language' in item:
                    lang = item['language']
                    stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
                
                if 'topics' in item:
                    for topic in item['topics']:
                        stats['topic_distribution'][topic] = stats['topic_distribution'].get(topic, 0) + 1
        
        elif data_type == 'design_files':
            stats['file_type_distribution'] = {}
            
            for item in data:
                if 'file_type' in item:
                    file_type = item['file_type']
                    stats['file_type_distribution'][file_type] = stats['file_type_distribution'].get(file_type, 0) + 1
        
        elif data_type == 'queries':
            stats['design_type_distribution'] = {}
            stats['constraint_distribution'] = {}
            
            for item in data:
                if 'design_type' in item:
                    design_type = item['design_type']
                    stats['design_type_distribution'][design_type] = stats['design_type_distribution'].get(design_type, 0) + 1
                
                if 'constraints' in item:
                    for constraint in item['constraints']:
                        stats['constraint_distribution'][constraint] = stats['constraint_distribution'].get(constraint, 0) + 1
        
        # 保存统计报告
        stats_file = self.data_dir / f'{data_type}_statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计报告已生成: {stats_file}")

def main():
    """主函数"""
    # 配置
    config = {
        'data_dir': 'data/real',
        'github_search_terms': ['RISC-V', 'DSP', 'ASIC', 'FPGA', 'chip design'],
        'local_directories': ['/path/to/local/designs']
    }
    
    # 初始化收集器
    collector = SimpleDataCollector(config)
    
    # 收集数据
    logger.info("开始收集真实数据...")
    
    # 1. 从GitHub收集
    github_data = collector.collect_from_github(
        config['github_search_terms'], 
        max_repos=5
    )
    
    if github_data:
        collector.save_collected_data(github_data, 'github_data')
    
    # 2. 从本地文件收集（如果有本地目录）
    for local_dir in config['local_directories']:
        if Path(local_dir).exists():
            local_data = collector.collect_from_local_files(local_dir)
            if local_data:
                collector.save_collected_data(local_data, 'local_data')
    
    # 3. 生成示例查询
    sample_queries = collector.generate_sample_queries(50)
    collector.save_collected_data(sample_queries, 'sample_queries')
    
    logger.info("真实数据收集完成！")

if __name__ == '__main__':
    main() 