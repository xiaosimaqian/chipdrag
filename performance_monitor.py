#!/usr/bin/env python3
"""
ChipDRAG OpenROAD并行执行性能监控工具
"""
import os
import sys
import time
import json
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval: int = 5):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.performance_data = []
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"✅ 性能监控已启动，监控间隔: {self.monitor_interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        print("✅ 性能监控已停止")
        return self.generate_report()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统资源信息
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # 统计OpenROAD进程
                openroad_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        if 'openroad' in proc.info['name'].lower():
                            openroad_processes.append({
                                'pid': proc.info['pid'],
                                'cpu_percent': proc.info['cpu_percent'],
                                'memory_percent': proc.info['memory_percent']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # 统计Python进程（实验脚本）
                python_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
                    try:
                        if 'python' in proc.info['name'].lower():
                            cmdline = proc.info['cmdline']
                            if cmdline and any('experiment.py' in arg for arg in cmdline):
                                python_processes.append({
                                    'pid': proc.info['pid'],
                                    'cpu_percent': proc.info['cpu_percent'],
                                    'memory_percent': proc.info['memory_percent']
                                })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # 记录性能数据
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent_total': cpu_percent,
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_percent': memory.percent,
                    'openroad_processes': openroad_processes,
                    'python_processes': python_processes,
                    'openroad_count': len(openroad_processes),
                    'python_count': len(python_processes)
                }
                
                self.performance_data.append(data_point)
                
                # 实时输出简要信息
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | "
                      f"CPU: {cpu_percent:.1f}% | "
                      f"内存: {memory.percent:.1f}% | "
                      f"OpenROAD进程: {len(openroad_processes)} | "
                      f"实验进程: {len(python_processes)}")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"❌ 监控异常: {e}")
                break
    
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.performance_data:
            return {}
        
        # 计算统计信息
        cpu_values = [d['cpu_percent_total'] for d in self.performance_data]
        memory_values = [d['memory_percent'] for d in self.performance_data]
        openroad_counts = [d['openroad_count'] for d in self.performance_data]
        
        report = {
            'monitoring_duration': str(datetime.now() - self.start_time) if self.start_time else "0",
            'total_data_points': len(self.performance_data),
            'cpu_utilization': {
                'average': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_utilization': {
                'average': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'openroad_parallelism': {
                'average_processes': sum(openroad_counts) / len(openroad_counts),
                'max_processes': max(openroad_counts),
                'min_processes': min(openroad_counts)
            },
            'hardware_info': {
                'cpu_cores': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        # 保存详细数据
        report_file = Path("performance_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'summary': report,
                'detailed_data': self.performance_data
            }, f, indent=2)
        
        print(f"\n📊 性能报告已生成: {report_file}")
        self._print_report_summary(report)
        
        return report
    
    def _print_report_summary(self, report: Dict[str, Any]):
        """打印报告摘要"""
        print("\n" + "="*60)
        print("📊 ChipDRAG OpenROAD并行执行性能报告")
        print("="*60)
        print(f"监控时长: {report['monitoring_duration']}")
        print(f"数据点数: {report['total_data_points']}")
        print(f"硬件配置: {report['hardware_info']['cpu_cores']}核CPU, {report['hardware_info']['memory_total_gb']:.1f}GB内存")
        print()
        print("CPU利用率:")
        print(f"  平均: {report['cpu_utilization']['average']:.1f}%")
        print(f"  最高: {report['cpu_utilization']['max']:.1f}%")
        print(f"  最低: {report['cpu_utilization']['min']:.1f}%")
        print()
        print("内存利用率:")
        print(f"  平均: {report['memory_utilization']['average']:.1f}%")
        print(f"  最高: {report['memory_utilization']['max']:.1f}%")
        print(f"  最低: {report['memory_utilization']['min']:.1f}%")
        print()
        print("OpenROAD并行度:")
        print(f"  平均进程数: {report['openroad_parallelism']['average_processes']:.1f}")
        print(f"  最大进程数: {report['openroad_parallelism']['max_processes']}")
        print(f"  最小进程数: {report['openroad_parallelism']['min_processes']}")
        print("="*60)

def main():
    """主函数"""
    print("🚀 ChipDRAG OpenROAD并行执行性能监控工具")
    print("使用方法:")
    print("1. 启动监控: python performance_monitor.py")
    print("2. 在另一个终端运行实验: python experiment.py --mode server")
    print("3. 按 Ctrl+C 停止监控并生成报告")
    print()
    
    monitor = PerformanceMonitor(monitor_interval=3)
    
    try:
        monitor.start_monitoring()
        
        print("💡 监控已启动，按 Ctrl+C 停止...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n📋 正在生成性能报告...")
        monitor.stop_monitoring()
        print("✅ 监控完成")

if __name__ == "__main__":
    main() 