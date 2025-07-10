#!/usr/bin/env python3
"""
ChipDRAG服务器监控脚本

功能：
1. 监控系统资源使用情况
2. 监控实验进度
3. 监控服务状态
4. 自动告警和恢复
"""

import os
import sys
import time
import json
import psutil
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import threading
import signal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/chipdrag/logs/monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChipDRAGMonitor:
    """ChipDRAG系统监控器"""
    
    def __init__(self, config_path: str = "server_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.alert_history = []
        
        # 监控阈值
        self.thresholds = self.config.get("services", {}).get("monitoring", {}).get("alert_threshold", {
            "cpu": 90,
            "memory": 85,
            "disk": 80
        })
        
        # 项目目录
        self.project_dir = Path(self.config.get("system", {}).get("project_dir", "/opt/chipdrag"))
        self.results_dir = self.project_dir / "results"
        self.logs_dir = self.project_dir / "logs"
        
        logger.info("ChipDRAG监控器初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def monitor_system_resources(self) -> Dict[str, Any]:
        """监控系统资源"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 检查网络连接
        network = psutil.net_io_counters()
        
        # 检查进程状态
        chipdrag_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'chipdrag' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                    chipdrag_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        resources = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage_percent': cpu_percent,
                'cores': psutil.cpu_count(),
                'load_avg': os.getloadavg()
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'free': memory.free,
                'usage_percent': memory.percent,
                'available': memory.available
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'usage_percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'processes': chipdrag_processes
        }
        
        # 检查告警
        self._check_alerts(resources)
        
        return resources
    
    def monitor_docker_containers(self) -> Dict[str, Any]:
        """监控Docker容器"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'],
                capture_output=True,
                text=True
            )
            
            containers = []
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container = json.loads(line)
                            containers.append(container)
                        except json.JSONDecodeError:
                            pass
            
            return {
                'timestamp': datetime.now().isoformat(),
                'containers': containers,
                'total_count': len(containers)
            }
        except Exception as e:
            logger.error(f"监控Docker容器失败: {e}")
            return {}
    
    def monitor_experiment_progress(self) -> Dict[str, Any]:
        """监控实验进度"""
        progress = {
            'timestamp': datetime.now().isoformat(),
            'active_experiments': [],
            'completed_experiments': [],
            'failed_experiments': []
        }
        
        # 扫描结果目录
        if self.results_dir.exists():
            for exp_dir in self.results_dir.iterdir():
                if exp_dir.is_dir():
                    exp_info = self._analyze_experiment_dir(exp_dir)
                    if exp_info['status'] == 'running':
                        progress['active_experiments'].append(exp_info)
                    elif exp_info['status'] == 'completed':
                        progress['completed_experiments'].append(exp_info)
                    elif exp_info['status'] == 'failed':
                        progress['failed_experiments'].append(exp_info)
        
        return progress
    
    def _analyze_experiment_dir(self, exp_dir: Path) -> Dict[str, Any]:
        """分析实验目录状态"""
        exp_info = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'status': 'unknown',
            'start_time': None,
            'end_time': None,
            'duration': None,
            'files': []
        }
        
        # 检查文件状态
        for file_path in exp_dir.rglob('*'):
            if file_path.is_file():
                exp_info['files'].append({
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        # 判断实验状态
        if (exp_dir / 'experiment_results.json').exists():
            exp_info['status'] = 'completed'
        elif (exp_dir / 'error.log').exists():
            exp_info['status'] = 'failed'
        elif any(f.name.endswith('.log') for f in exp_dir.rglob('*.log')):
            exp_info['status'] = 'running'
        
        return exp_info
    
    def _check_alerts(self, resources: Dict[str, Any]):
        """检查告警条件"""
        alerts = []
        
        # CPU告警
        if resources['cpu']['usage_percent'] > self.thresholds['cpu']:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"CPU使用率过高: {resources['cpu']['usage_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # 内存告警
        if resources['memory']['usage_percent'] > self.thresholds['memory']:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"内存使用率过高: {resources['memory']['usage_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # 磁盘告警
        if resources['disk']['usage_percent'] > self.thresholds['disk']:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"磁盘使用率过高: {resources['disk']['usage_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # 记录告警
        for alert in alerts:
            logger.warning(alert['message'])
            self.alert_history.append(alert)
    
    def generate_status_report(self) -> Dict[str, Any]:
        """生成状态报告"""
        resources = self.monitor_system_resources()
        containers = self.monitor_docker_containers()
        experiments = self.monitor_experiment_progress()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_resources': resources,
            'docker_containers': containers,
            'experiments': experiments,
            'alerts': self.alert_history[-10:],  # 最近10条告警
            'summary': {
                'total_experiments': len(experiments['active_experiments']) + 
                                   len(experiments['completed_experiments']) + 
                                   len(experiments['failed_experiments']),
                'active_experiments': len(experiments['active_experiments']),
                'system_healthy': len([a for a in self.alert_history[-10:] if a['level'] == 'warning']) == 0
            }
        }
    
    def start_monitoring(self, interval: int = 30):
        """启动监控"""
        self.running = True
        logger.info(f"开始监控，间隔: {interval}秒")
        
        while self.running:
            try:
                # 生成状态报告
                report = self.generate_status_report()
                
                # 保存报告
                report_file = self.logs_dir / f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                # 打印简要状态
                print(f"\n=== ChipDRAG状态报告 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                print(f"CPU使用率: {report['system_resources']['cpu']['usage_percent']:.1f}%")
                print(f"内存使用率: {report['system_resources']['memory']['usage_percent']:.1f}%")
                print(f"磁盘使用率: {report['system_resources']['disk']['usage_percent']:.1f}%")
                print(f"活跃实验: {report['summary']['active_experiments']}个")
                print(f"Docker容器: {report['docker_containers']['total_count']}个")
                print(f"系统健康: {'✓' if report['summary']['system_healthy'] else '✗'}")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("收到停止信号，正在关闭监控...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"监控过程中发生错误: {e}")
                time.sleep(interval)
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        logger.info("监控已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChipDRAG服务器监控')
    parser.add_argument('--config', default='server_config.json', help='配置文件路径')
    parser.add_argument('--interval', type=int, default=30, help='监控间隔(秒)')
    parser.add_argument('--once', action='store_true', help='只运行一次')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = ChipDRAGMonitor(args.config)
    
    # 设置信号处理
    def signal_handler(signum, frame):
        logger.info(f"收到信号 {signum}，正在停止监控...")
        monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.once:
        # 只运行一次
        report = monitor.generate_status_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        # 持续监控
        monitor.start_monitoring(args.interval)

if __name__ == "__main__":
    main() 