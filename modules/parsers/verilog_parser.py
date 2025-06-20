import re
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def clean_verilog_content(content: str) -> str:
    """清理Verilog文件内容，移除注释和多余空白
    
    Args:
        content: Verilog文件内容
        
    Returns:
        清理后的内容
    """
    # 移除单行注释
    content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
    # 移除多行注释
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # 移除多余空白
    content = re.sub(r'\s+', ' ', content)
    return content

def parse_verilog_file(file_path: str) -> Dict[str, Any]:
    """解析Verilog文件
    
    Args:
        file_path: Verilog文件路径
        
    Returns:
        解析后的数据字典
    """
    logger.info(f"开始解析Verilog文件: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 清理文件内容
        content = clean_verilog_content(content)
        
        # 解析模块
        modules = []
        module_pattern = r'module\s+(\w+)\s*\((.*?)\);'
        module_matches = re.finditer(module_pattern, content, re.DOTALL)
        
        for module_match in module_matches:
            module_name = module_match.group(1)
            port_list = module_match.group(2)
            
            # 解析端口列表
            ports = []
            port_names = [p.strip() for p in port_list.split(',')]
            
            # 查找端口声明
            port_decl_pattern = r'(input|output|inout)\s+(\w+)'
            port_decls = re.finditer(port_decl_pattern, content)
            port_types = {}
            for decl in port_decls:
                port_type = decl.group(1)
                port_name = decl.group(2)
                port_types[port_name] = port_type
            
            # 为每个端口创建端口对象
            for port_name in port_names:
                if port_name in port_types:
                    ports.append({
                        'name': port_name,
                        'direction': port_types[port_name]
                    })
            
            # 解析实例和网络
            instances = []
            nets = set()  # 使用集合来存储唯一的网络名称
            
            instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\);'
            instance_matches = re.finditer(instance_pattern, content)
            
            for instance_match in instance_matches:
                instance_type = instance_match.group(1)
                instance_name = instance_match.group(2)
                port_connections = instance_match.group(3)
                
                # 解析端口连接
                connections = {}
                for conn in port_connections.split(','):
                    conn = conn.strip()
                    if '.' in conn:
                        port, net = conn.split('.')
                        port = port.strip()
                        net = net.strip()
                        connections[port] = net
                        # 将连接的网络添加到网络集合中
                        nets.add(net)
                
                instances.append({
                    'type': instance_type,
                    'name': instance_name,
                    'connections': connections
                })
            
            # 将网络集合转换为列表
            nets_list = [{'name': net, 'type': 'wire'} for net in nets]
            
            modules.append({
                'name': module_name,
                'ports': ports,
                'instances': instances,
                'nets': nets_list
            })
        
        result = {'modules': modules}
        logger.info(f"Verilog文件解析完成，找到 {len(result['modules'])} 个模块")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"Verilog文件不存在: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"解析Verilog文件时出错: {str(e)}")
        raise

class VerilogParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """解析Verilog文件
        
        Args:
            file_path: Verilog文件路径
            
        Returns:
            解析后的数据字典
        """
        try:
            self.logger.info(f"开始解析Verilog文件: {file_path}")
            result = parse_verilog_file(file_path)
            self.logger.info(f"Verilog文件解析完成，找到 {len(result['modules'])} 个模块")
            return result
        except Exception as e:
            self.logger.error(f"解析Verilog文件时出错: {str(e)}")
            raise