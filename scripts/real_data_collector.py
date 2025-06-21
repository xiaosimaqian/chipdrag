#!/usr/bin/env python3
"""
真实数据收集器
收集芯片设计领域的真实数据用于Chip-D-RAG系统训练
"""

import json
import logging
import sqlite3
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import pandas as pd
from dataclasses import dataclass, asdict
import os
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DesignQuery:
    """设计查询数据"""
    query_id: str
    query_text: str
    design_type: str
    constraints: List[str]
    complexity: float
    user_id: str
    timestamp: str
    source: str

@dataclass
class DesignInfo:
    """设计信息数据"""
    design_id: str
    design_type: str
    technology_node: str
    area_constraint: float
    power_budget: float
    timing_constraint: float
    constraints: List[Dict[str, Any]]
    netlist_file: str
    def_file: str
    timestamp: str
    source: str

@dataclass
class LayoutResult:
    """布局结果数据"""
    result_id: str
    query_id: str
    design_id: str
    layout_file: str
    wirelength: float
    congestion: float
    timing_score: float
    power_score: float
    area_utilization: float
    generation_time: float
    method: str
    timestamp: str

@dataclass
class QualityFeedback:
    """质量反馈数据"""
    feedback_id: str
    result_id: str
    user_id: str
    wirelength_score: float
    congestion_score: float
    timing_score: float
    power_score: float
    overall_score: float
    feedback_text: str
    timestamp: str

class RealDataCollector:
    """真实数据收集器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/real'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self.db_path = self.data_dir / 'real_data.db'
        self._init_database()
        
        # 数据源配置
        self.sources = config.get('sources', {})
        
        logger.info(f"真实数据收集器初始化完成，数据目录: {self.data_dir}")
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建查询表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                design_type TEXT NOT NULL,
                constraints TEXT NOT NULL,
                complexity REAL NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL
            )
        ''')
        
        # 创建设计信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS design_info (
                design_id TEXT PRIMARY KEY,
                design_type TEXT NOT NULL,
                technology_node TEXT NOT NULL,
                area_constraint REAL NOT NULL,
                power_budget REAL NOT NULL,
                timing_constraint REAL NOT NULL,
                constraints TEXT NOT NULL,
                netlist_file TEXT,
                def_file TEXT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL
            )
        ''')
        
        # 创建布局结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS layout_results (
                result_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL,
                design_id TEXT NOT NULL,
                layout_file TEXT,
                wirelength REAL,
                congestion REAL,
                timing_score REAL,
                power_score REAL,
                area_utilization REAL,
                generation_time REAL,
                method TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (query_id) REFERENCES design_queries (query_id),
                FOREIGN KEY (design_id) REFERENCES design_info (design_id)
            )
        ''')
        
        # 创建质量反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_feedback (
                feedback_id TEXT PRIMARY KEY,
                result_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                wirelength_score REAL NOT NULL,
                congestion_score REAL NOT NULL,
                timing_score REAL NOT NULL,
                power_score REAL NOT NULL,
                overall_score REAL NOT NULL,
                feedback_text TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (result_id) REFERENCES layout_results (result_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("数据库初始化完成")
    
    def collect_from_eda_tools(self, tool_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从EDA工具收集数据
        
        Args:
            tool_config: EDA工具配置
            
        Returns:
            List[Dict[str, Any]]: 收集的数据
        """
        logger.info("开始从EDA工具收集数据...")
        
        collected_data = []
        tool_type = tool_config.get('type', 'unknown')
        
        if tool_type == 'synopsys':
            collected_data.extend(self._collect_from_synopsys(tool_config))
        elif tool_type == 'cadence':
            collected_data.extend(self._collect_from_cadence(tool_config))
        elif tool_type == 'mentor':
            collected_data.extend(self._collect_from_mentor(tool_config))
        else:
            logger.warning(f"不支持的EDA工具类型: {tool_type}")
        
        logger.info(f"从EDA工具收集到 {len(collected_data)} 条数据")
        return collected_data
    
    def _collect_from_synopsys(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Synopsys工具收集数据"""
        data = []
        
        # 读取Design Compiler日志
        dc_log_dir = Path(config.get('dc_log_dir', ''))
        if dc_log_dir.exists():
            for log_file in dc_log_dir.glob('*.log'):
                data.extend(self._parse_dc_log(log_file))
        
        # 读取IC Compiler日志
        ic_log_dir = Path(config.get('ic_log_dir', ''))
        if ic_log_dir.exists():
            for log_file in ic_log_dir.glob('*.log'):
                data.extend(self._parse_ic_log(log_file))
        
        return data
    
    def _collect_from_cadence(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Cadence工具收集数据"""
        data = []
        
        # 读取Innovus日志
        innovus_log_dir = Path(config.get('innovus_log_dir', ''))
        if innovus_log_dir.exists():
            for log_file in innovus_log_dir.glob('*.log'):
                data.extend(self._parse_innovus_log(log_file))
        
        return data
    
    def _collect_from_mentor(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Mentor工具收集数据"""
        data = []
        
        # 读取Calibre日志
        calibre_log_dir = Path(config.get('calibre_log_dir', ''))
        if calibre_log_dir.exists():
            for log_file in calibre_log_dir.glob('*.log'):
                data.extend(self._parse_calibre_log(log_file))
        
        return data
    
    def _parse_dc_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """解析Design Compiler日志"""
        data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取设计信息
            design_info = self._extract_design_info_from_log(content)
            if design_info:
                data.append(design_info)
            
        except Exception as e:
            logger.error(f"解析DC日志失败 {log_file}: {str(e)}")
        
        return data
    
    def _parse_ic_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """解析IC Compiler日志"""
        data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取布局结果
            layout_results = self._extract_layout_results_from_log(content)
            data.extend(layout_results)
            
        except Exception as e:
            logger.error(f"解析IC日志失败 {log_file}: {str(e)}")
        
        return data
    
    def _parse_innovus_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """解析Innovus日志"""
        data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取布局结果
            layout_results = self._extract_layout_results_from_log(content)
            data.extend(layout_results)
            
        except Exception as e:
            logger.error(f"解析Innovus日志失败 {log_file}: {str(e)}")
        
        return data
    
    def _parse_calibre_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """解析Calibre日志"""
        data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取DRC/LVS结果
            verification_results = self._extract_verification_results_from_log(content)
            data.extend(verification_results)
            
        except Exception as e:
            logger.error(f"解析Calibre日志失败 {log_file}: {str(e)}")
        
        return data
    
    def _extract_design_info_from_log(self, content: str) -> Optional[Dict[str, Any]]:
        """从日志中提取设计信息"""
        try:
            # 这里需要根据具体的日志格式进行解析
            # 示例解析逻辑
            design_info = {
                'design_type': 'unknown',
                'technology_node': 'unknown',
                'area_constraint': 0.0,
                'power_budget': 0.0,
                'timing_constraint': 0.0,
                'constraints': []
            }
            
            # 解析技术节点
            if 'technology' in content.lower():
                # 提取技术节点信息
                pass
            
            # 解析约束信息
            if 'constraint' in content.lower():
                # 提取约束信息
                pass
            
            return design_info
            
        except Exception as e:
            logger.error(f"提取设计信息失败: {str(e)}")
            return None
    
    def _extract_layout_results_from_log(self, content: str) -> List[Dict[str, Any]]:
        """从日志中提取布局结果"""
        results = []
        
        try:
            # 解析布局质量指标
            if 'wirelength' in content.lower():
                # 提取布线长度
                pass
            
            if 'congestion' in content.lower():
                # 提取拥塞信息
                pass
            
            if 'timing' in content.lower():
                # 提取时序信息
                pass
            
            # 创建布局结果对象
            result = {
                'wirelength': 0.0,
                'congestion': 0.0,
                'timing_score': 0.0,
                'power_score': 0.0,
                'area_utilization': 0.0,
                'generation_time': 0.0,
                'method': 'eda_tool'
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"提取布局结果失败: {str(e)}")
        
        return results
    
    def _extract_verification_results_from_log(self, content: str) -> List[Dict[str, Any]]:
        """从日志中提取验证结果"""
        results = []
        
        try:
            # 解析DRC/LVS结果
            if 'drc' in content.lower():
                # 提取DRC结果
                pass
            
            if 'lvs' in content.lower():
                # 提取LVS结果
                pass
            
        except Exception as e:
            logger.error(f"提取验证结果失败: {str(e)}")
        
        return results
    
    def collect_from_design_repositories(self, repo_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从设计仓库收集数据
        
        Args:
            repo_config: 仓库配置
            
        Returns:
            List[Dict[str, Any]]: 收集的数据
        """
        logger.info("开始从设计仓库收集数据...")
        
        collected_data = []
        repo_type = repo_config.get('type', 'unknown')
        
        if repo_type == 'github':
            collected_data.extend(self._collect_from_github(repo_config))
        elif repo_type == 'gitlab':
            collected_data.extend(self._collect_from_gitlab(repo_config))
        elif repo_type == 'local':
            collected_data.extend(self._collect_from_local_repo(repo_config))
        else:
            logger.warning(f"不支持的仓库类型: {repo_type}")
        
        logger.info(f"从设计仓库收集到 {len(collected_data)} 条数据")
        return collected_data
    
    def _collect_from_github(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从GitHub收集数据"""
        data = []
        
        # GitHub API配置
        token = config.get('token', '')
        repos = config.get('repos', [])
        
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        for repo in repos:
            try:
                # 获取仓库内容
                api_url = f"https://api.github.com/repos/{repo}/contents"
                response = requests.get(api_url, headers=headers)
                
                if response.status_code == 200:
                    contents = response.json()
                    
                    # 查找设计文件
                    for item in contents:
                        if item['type'] == 'file':
                            if self._is_design_file(item['name']):
                                file_data = self._download_and_parse_file(
                                    item['download_url'], item['name']
                                )
                                if file_data:
                                    data.append(file_data)
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"从GitHub收集数据失败 {repo}: {str(e)}")
        
        return data
    
    def _collect_from_gitlab(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从GitLab收集数据"""
        data = []
        
        # GitLab API配置
        token = config.get('token', '')
        base_url = config.get('base_url', 'https://gitlab.com')
        projects = config.get('projects', [])
        
        headers = {
            'PRIVATE-TOKEN': token
        }
        
        for project in projects:
            try:
                # 获取项目文件
                api_url = f"{base_url}/api/v4/projects/{project}/repository/tree"
                response = requests.get(api_url, headers=headers)
                
                if response.status_code == 200:
                    contents = response.json()
                    
                    # 查找设计文件
                    for item in contents:
                        if item['type'] == 'blob':
                            if self._is_design_file(item['name']):
                                file_data = self._download_and_parse_file(
                                    item['path'], item['name']
                                )
                                if file_data:
                                    data.append(file_data)
                
                # 避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"从GitLab收集数据失败 {project}: {str(e)}")
        
        return data
    
    def _collect_from_local_repo(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从本地仓库收集数据"""
        data = []
        
        repo_path = Path(config.get('repo_path', ''))
        if not repo_path.exists():
            logger.error(f"本地仓库路径不存在: {repo_path}")
            return data
        
        # 查找设计文件
        design_extensions = ['.v', '.vhd', '.def', '.lef', '.lib', '.sdc', '.tcl']
        
        for ext in design_extensions:
            for file_path in repo_path.rglob(f'*{ext}'):
                try:
                    file_data = self._parse_design_file(file_path)
                    if file_data:
                        data.append(file_data)
                except Exception as e:
                    logger.error(f"解析设计文件失败 {file_path}: {str(e)}")
        
        return data
    
    def _is_design_file(self, filename: str) -> bool:
        """判断是否为设计文件"""
        design_extensions = ['.v', '.vhd', '.def', '.lef', '.lib', '.sdc', '.tcl', '.gds']
        return any(filename.lower().endswith(ext) for ext in design_extensions)
    
    def _download_and_parse_file(self, url: str, filename: str) -> Optional[Dict[str, Any]]:
        """下载并解析文件"""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # 保存文件
                file_path = self.data_dir / 'downloads' / filename
                file_path.parent.mkdir(exist_ok=True)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # 解析文件
                return self._parse_design_file(file_path)
            
        except Exception as e:
            logger.error(f"下载文件失败 {url}: {str(e)}")
        
        return None
    
    def _parse_design_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析设计文件"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.v':
                return self._parse_verilog_file(file_path)
            elif file_ext == '.vhd':
                return self._parse_vhdl_file(file_path)
            elif file_ext == '.def':
                return self._parse_def_file(file_path)
            elif file_ext == '.lef':
                return self._parse_lef_file(file_path)
            elif file_ext == '.lib':
                return self._parse_lib_file(file_path)
            elif file_ext == '.sdc':
                return self._parse_sdc_file(file_path)
            elif file_ext == '.tcl':
                return self._parse_tcl_file(file_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"解析设计文件失败 {file_path}: {str(e)}")
            return None
    
    def _parse_verilog_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析Verilog文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取模块信息
            module_info = {
                'file_type': 'verilog',
                'file_path': str(file_path),
                'module_name': self._extract_module_name(content),
                'port_count': self._extract_port_count(content),
                'instance_count': self._extract_instance_count(content),
                'timestamp': datetime.now().isoformat()
            }
            
            return module_info
            
        except Exception as e:
            logger.error(f"解析Verilog文件失败 {file_path}: {str(e)}")
            return None
    
    def _parse_vhdl_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析VHDL文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取实体信息
            entity_info = {
                'file_type': 'vhdl',
                'file_path': str(file_path),
                'entity_name': self._extract_entity_name(content),
                'port_count': self._extract_vhdl_port_count(content),
                'timestamp': datetime.now().isoformat()
            }
            
            return entity_info
            
        except Exception as e:
            logger.error(f"解析VHDL文件失败 {file_path}: {str(e)}")
            return None
    
    def _parse_def_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """解析DEF文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取布局信息
            layout_info = {
                'file_type': 'def',
                'file_path': str(file_path),
                'design_name': self._extract_def_design_name(content),
                'component_count': self._extract_component_count(content),
                'net_count': self._extract_net_count(content),
                'timestamp': datetime.now().isoformat()
            }
            
            return layout_info
            
        except Exception as e:
            logger.error(f"解析DEF文件失败 {file_path}: {str(e)}")
            return None
    
    def _extract_module_name(self, content: str) -> str:
        """提取模块名"""
        import re
        match = re.search(r'module\s+(\w+)', content)
        return match.group(1) if match else 'unknown'
    
    def _extract_port_count(self, content: str) -> int:
        """提取端口数量"""
        import re
        ports = re.findall(r'(input|output|inout)\s+', content)
        return len(ports)
    
    def _extract_instance_count(self, content: str) -> int:
        """提取实例数量"""
        import re
        instances = re.findall(r'(\w+)\s+(\w+)\s*\(', content)
        return len(instances)
    
    def _extract_entity_name(self, content: str) -> str:
        """提取实体名"""
        import re
        match = re.search(r'entity\s+(\w+)', content, re.IGNORECASE)
        return match.group(1) if match else 'unknown'
    
    def _extract_vhdl_port_count(self, content: str) -> int:
        """提取VHDL端口数量"""
        import re
        ports = re.findall(r'(\w+)\s*:\s*(in|out|inout)', content, re.IGNORECASE)
        return len(ports)
    
    def _extract_def_design_name(self, content: str) -> str:
        """提取DEF设计名"""
        import re
        match = re.search(r'DESIGN\s+(\w+)', content)
        return match.group(1) if match else 'unknown'
    
    def _extract_component_count(self, content: str) -> int:
        """提取组件数量"""
        import re
        components = re.findall(r'COMPONENTS\s+(\d+)', content)
        return int(components[0]) if components else 0
    
    def _extract_net_count(self, content: str) -> int:
        """提取网络数量"""
        import re
        nets = re.findall(r'NETS\s+(\d+)', content)
        return int(nets[0]) if nets else 0
    
    def collect_from_user_interactions(self, interaction_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从用户交互收集数据
        
        Args:
            interaction_config: 交互配置
            
        Returns:
            List[Dict[str, Any]]: 收集的数据
        """
        logger.info("开始从用户交互收集数据...")
        
        collected_data = []
        
        # 收集查询数据
        queries = self._collect_user_queries(interaction_config)
        collected_data.extend(queries)
        
        # 收集反馈数据
        feedbacks = self._collect_user_feedback(interaction_config)
        collected_data.extend(feedbacks)
        
        logger.info(f"从用户交互收集到 {len(collected_data)} 条数据")
        return collected_data
    
    def _collect_user_queries(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集用户查询"""
        queries = []
        
        # 从日志文件收集
        log_dir = Path(config.get('log_dir', ''))
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                queries.extend(self._parse_query_log(log_file))
        
        # 从数据库收集
        db_path = config.get('db_path', '')
        if db_path and Path(db_path).exists():
            queries.extend(self._collect_queries_from_db(db_path))
        
        return queries
    
    def _collect_user_feedback(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集用户反馈"""
        feedbacks = []
        
        # 从反馈文件收集
        feedback_dir = Path(config.get('feedback_dir', ''))
        if feedback_dir.exists():
            for feedback_file in feedback_dir.glob('*.json'):
                feedbacks.extend(self._parse_feedback_file(feedback_file))
        
        return feedbacks
    
    def _parse_query_log(self, log_file: Path) -> List[Dict[str, Any]]:
        """解析查询日志"""
        queries = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'query' in line.lower():
                        query_data = self._extract_query_from_log_line(line)
                        if query_data:
                            queries.append(query_data)
        except Exception as e:
            logger.error(f"解析查询日志失败 {log_file}: {str(e)}")
        
        return queries
    
    def _extract_query_from_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """从日志行提取查询信息"""
        try:
            # 这里需要根据具体的日志格式进行解析
            # 示例解析逻辑
            if 'query' in line.lower():
                return {
                    'query_text': line.strip(),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'log_file'
                }
        except Exception as e:
            logger.error(f"提取查询信息失败: {str(e)}")
        
        return None
    
    def _parse_feedback_file(self, feedback_file: Path) -> List[Dict[str, Any]]:
        """解析反馈文件"""
        feedbacks = []
        
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    feedbacks.extend(data)
                else:
                    feedbacks.append(data)
        except Exception as e:
            logger.error(f"解析反馈文件失败 {feedback_file}: {str(e)}")
        
        return feedbacks
    
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
        
        # 保存到数据库
        self._save_to_database(data, data_type)
        
        logger.info(f"数据已保存到 {output_file}")
    
    def _save_to_database(self, data: List[Dict[str, Any]], data_type: str):
        """保存数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if data_type == 'queries':
                for item in data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO design_queries 
                        (query_id, query_text, design_type, constraints, complexity, user_id, timestamp, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.get('query_id', hashlib.md5(item['query_text'].encode()).hexdigest()),
                        item['query_text'],
                        item.get('design_type', 'unknown'),
                        json.dumps(item.get('constraints', [])),
                        item.get('complexity', 0.5),
                        item.get('user_id', 'unknown'),
                        item.get('timestamp', datetime.now().isoformat()),
                        item.get('source', 'unknown')
                    ))
            
            elif data_type == 'designs':
                for item in data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO design_info 
                        (design_id, design_type, technology_node, area_constraint, power_budget, timing_constraint, constraints, netlist_file, def_file, timestamp, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.get('design_id', hashlib.md5(str(item).encode()).hexdigest()),
                        item.get('design_type', 'unknown'),
                        item.get('technology_node', 'unknown'),
                        item.get('area_constraint', 0.0),
                        item.get('power_budget', 0.0),
                        item.get('timing_constraint', 0.0),
                        json.dumps(item.get('constraints', [])),
                        item.get('netlist_file', ''),
                        item.get('def_file', ''),
                        item.get('timestamp', datetime.now().isoformat()),
                        item.get('source', 'unknown')
                    ))
            
            elif data_type == 'results':
                for item in data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO layout_results 
                        (result_id, query_id, design_id, layout_file, wirelength, congestion, timing_score, power_score, area_utilization, generation_time, method, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.get('result_id', hashlib.md5(str(item).encode()).hexdigest()),
                        item.get('query_id', 'unknown'),
                        item.get('design_id', 'unknown'),
                        item.get('layout_file', ''),
                        item.get('wirelength', 0.0),
                        item.get('congestion', 0.0),
                        item.get('timing_score', 0.0),
                        item.get('power_score', 0.0),
                        item.get('area_utilization', 0.0),
                        item.get('generation_time', 0.0),
                        item.get('method', 'unknown'),
                        item.get('timestamp', datetime.now().isoformat())
                    ))
            
            elif data_type == 'feedback':
                for item in data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO quality_feedback 
                        (feedback_id, result_id, user_id, wirelength_score, congestion_score, timing_score, power_score, overall_score, feedback_text, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item.get('feedback_id', hashlib.md5(str(item).encode()).hexdigest()),
                        item.get('result_id', 'unknown'),
                        item.get('user_id', 'unknown'),
                        item.get('wirelength_score', 0.0),
                        item.get('congestion_score', 0.0),
                        item.get('timing_score', 0.0),
                        item.get('power_score', 0.0),
                        item.get('overall_score', 0.0),
                        item.get('feedback_text', ''),
                        item.get('timestamp', datetime.now().isoformat())
                    ))
            
            conn.commit()
            logger.info(f"数据已保存到数据库，类型: {data_type}")
            
        except Exception as e:
            logger.error(f"保存数据到数据库失败: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    def generate_data_report(self) -> Dict[str, Any]:
        """生成数据收集报告"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 统计各表数据量
            cursor.execute("SELECT COUNT(*) FROM design_queries")
            query_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM design_info")
            design_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM layout_results")
            result_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM quality_feedback")
            feedback_count = cursor.fetchone()[0]
            
            # 统计设计类型分布
            cursor.execute("SELECT design_type, COUNT(*) FROM design_info GROUP BY design_type")
            design_type_distribution = dict(cursor.fetchall())
            
            # 统计来源分布
            cursor.execute("SELECT source, COUNT(*) FROM design_queries GROUP BY source")
            source_distribution = dict(cursor.fetchall())
            
            report = {
                'data_summary': {
                    'total_queries': query_count,
                    'total_designs': design_count,
                    'total_results': result_count,
                    'total_feedback': feedback_count
                },
                'design_type_distribution': design_type_distribution,
                'source_distribution': source_distribution,
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # 保存报告
            report_file = self.data_dir / 'data_collection_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据收集报告已生成: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"生成数据报告失败: {str(e)}")
            return {}
        finally:
            conn.close()

def main():
    """主函数"""
    # 配置
    config = {
        'data_dir': 'data/real',
        'sources': {
            'eda_tools': {
                'synopsys': {
                    'type': 'synopsys',
                    'dc_log_dir': '/path/to/dc/logs',
                    'ic_log_dir': '/path/to/ic/logs'
                },
                'cadence': {
                    'type': 'cadence',
                    'innovus_log_dir': '/path/to/innovus/logs'
                }
            },
            'repositories': {
                'github': {
                    'type': 'github',
                    'token': 'your_github_token',
                    'repos': ['username/repo1', 'username/repo2']
                },
                'local': {
                    'type': 'local',
                    'repo_path': '/path/to/local/repo'
                }
            },
            'user_interactions': {
                'log_dir': '/path/to/logs',
                'feedback_dir': '/path/to/feedback'
            }
        }
    }
    
    # 初始化收集器
    collector = RealDataCollector(config)
    
    # 收集数据
    logger.info("开始收集真实数据...")
    
    # 从EDA工具收集
    eda_data = []
    for tool_name, tool_config in config['sources']['eda_tools'].items():
        data = collector.collect_from_eda_tools(tool_config)
        eda_data.extend(data)
    
    if eda_data:
        collector.save_collected_data(eda_data, 'eda_tools')
    
    # 从设计仓库收集
    repo_data = []
    for repo_name, repo_config in config['sources']['repositories'].items():
        data = collector.collect_from_design_repositories(repo_config)
        repo_data.extend(data)
    
    if repo_data:
        collector.save_collected_data(repo_data, 'repositories')
    
    # 从用户交互收集
    interaction_data = collector.collect_from_user_interactions(
        config['sources']['user_interactions']
    )
    
    if interaction_data:
        collector.save_collected_data(interaction_data, 'user_interactions')
    
    # 生成报告
    collector.generate_data_report()
    
    logger.info("真实数据收集完成！")

if __name__ == '__main__':
    main() 