#!/usr/bin/env python3
"""
工程清理脚本
清理各种无用文件和数据
"""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import List, Set
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProjectCleaner:
    """工程清理器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleaned_files = []
        self.cleaned_dirs = []
        self.total_size_saved = 0
        
    def clean_python_cache(self) -> int:
        """清理Python缓存文件"""
        logger.info("清理Python缓存文件...")
        count = 0
        
        # 清理__pycache__目录
        for cache_dir in self.project_root.rglob("__pycache__"):
            if cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    self.cleaned_dirs.append(str(cache_dir))
                    count += 1
                    logger.info(f"删除缓存目录: {cache_dir}")
                except Exception as e:
                    logger.warning(f"删除缓存目录失败 {cache_dir}: {e}")
        
        # 清理.pyc和.pyo文件
        for pyc_file in self.project_root.rglob("*.pyc"):
            try:
                size = pyc_file.stat().st_size
                pyc_file.unlink()
                self.cleaned_files.append(str(pyc_file))
                self.total_size_saved += size
                count += 1
            except Exception as e:
                logger.warning(f"删除.pyc文件失败 {pyc_file}: {e}")
        
        for pyo_file in self.project_root.rglob("*.pyo"):
            try:
                size = pyo_file.stat().st_size
                pyo_file.unlink()
                self.cleaned_files.append(str(pyo_file))
                self.total_size_saved += size
                count += 1
            except Exception as e:
                logger.warning(f"删除.pyo文件失败 {pyo_file}: {e}")
        
        logger.info(f"清理了 {count} 个Python缓存文件/目录")
        return count
    
    def clean_log_files(self) -> int:
        """清理日志文件"""
        logger.info("清理日志文件...")
        count = 0
        
        # 清理各种日志文件
        log_patterns = [
            "*.log",
            "*.out",
            "*.err",
            "*.trace",
            "*.debug"
        ]
        
        for pattern in log_patterns:
            for log_file in self.project_root.rglob(pattern):
                if log_file.is_file():
                    try:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        self.cleaned_files.append(str(log_file))
                        self.total_size_saved += size
                        count += 1
                    except Exception as e:
                        logger.warning(f"删除日志文件失败 {log_file}: {e}")
        
        logger.info(f"清理了 {count} 个日志文件")
        return count
    
    def clean_test_files(self) -> int:
        """清理测试和实验文件"""
        logger.info("清理测试和实验文件...")
        count = 0
        
        # 测试文件模式
        test_patterns = [
            "test_*.py",
            "*_test.py",
            "*_example.py",
            "*_demo.py",
            "*_validation*.py",
            "*_experiment*.py",
            "simple_*.py"
        ]
        
        for pattern in test_patterns:
            for test_file in self.project_root.rglob(pattern):
                if test_file.is_file() and test_file.name != "__init__.py":
                    try:
                        size = test_file.stat().st_size
                        test_file.unlink()
                        self.cleaned_files.append(str(test_file))
                        self.total_size_saved += size
                        count += 1
                        logger.info(f"删除测试文件: {test_file}")
                    except Exception as e:
                        logger.warning(f"删除测试文件失败 {test_file}: {e}")
        
        logger.info(f"清理了 {count} 个测试/实验文件")
        return count
    
    def clean_backup_files(self) -> int:
        """清理备份文件"""
        logger.info("清理备份文件...")
        count = 0
        
        backup_patterns = [
            "*.backup",
            "*.bak",
            "*_backup.py",
            "*_fixed.py",
            "*_old.py",
            "*~",
            ".#*"
        ]
        
        for pattern in backup_patterns:
            for backup_file in self.project_root.rglob(pattern):
                if backup_file.is_file():
                    try:
                        size = backup_file.stat().st_size
                        backup_file.unlink()
                        self.cleaned_files.append(str(backup_file))
                        self.total_size_saved += size
                        count += 1
                    except Exception as e:
                        logger.warning(f"删除备份文件失败 {backup_file}: {e}")
        
        logger.info(f"清理了 {count} 个备份文件")
        return count
    
    def clean_temp_files(self) -> int:
        """清理临时文件"""
        logger.info("清理临时文件...")
        count = 0
        
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.swp",
            "*.swo",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                if temp_file.is_file():
                    try:
                        size = temp_file.stat().st_size
                        temp_file.unlink()
                        self.cleaned_files.append(str(temp_file))
                        self.total_size_saved += size
                        count += 1
                    except Exception as e:
                        logger.warning(f"删除临时文件失败 {temp_file}: {e}")
        
        logger.info(f"清理了 {count} 个临时文件")
        return count
    
    def clean_invalid_results(self) -> int:
        """清理无效的实验结果"""
        logger.info("清理无效的实验结果...")
        count = 0
        
        # 清理包含Infinity或无效数据的JSON文件
        for json_file in self.project_root.rglob("*.json"):
            if json_file.is_file():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # 检查是否包含无效数据
                    if self._contains_invalid_data(data):
                        size = json_file.stat().st_size
                        json_file.unlink()
                        self.cleaned_files.append(str(json_file))
                        self.total_size_saved += size
                        count += 1
                        logger.info(f"删除无效结果文件: {json_file}")
                        
                except (json.JSONDecodeError, Exception) as e:
                    # 如果JSON文件损坏，也删除
                    try:
                        size = json_file.stat().st_size
                        json_file.unlink()
                        self.cleaned_files.append(str(json_file))
                        self.total_size_saved += size
                        count += 1
                        logger.info(f"删除损坏的JSON文件: {json_file}")
                    except Exception as e2:
                        logger.warning(f"删除JSON文件失败 {json_file}: {e2}")
        
        logger.info(f"清理了 {count} 个无效结果文件")
        return count
    
    def _contains_invalid_data(self, data) -> bool:
        """检查数据是否包含无效值"""
        if isinstance(data, dict):
            for key, value in data.items():
                if self._contains_invalid_data(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_invalid_data(item):
                    return True
        elif isinstance(data, (int, float)):
            # 检查是否为Infinity或NaN
            if data == float('inf') or data == float('-inf') or data != data:
                return True
        elif isinstance(data, str):
            # 检查字符串是否包含"Infinity"
            if "Infinity" in data or "inf" in data.lower():
                return True
        
        return False
    
    def clean_large_data_dirs(self) -> int:
        """清理大型数据目录"""
        logger.info("清理大型数据目录...")
        count = 0
        
        # 需要清理的大型数据目录
        large_dirs = [
            "data/processed",  # 2.2GB
            "data/cache",      # 12MB
            "temp",            # 100KB
            "results/experiments",
            "results/rl_optimization_experiment",
            "simple_enhanced_analysis",
            "enhanced_analysis"
        ]
        
        for dir_path in large_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                try:
                    # 计算目录大小
                    total_size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
                    
                    shutil.rmtree(full_path)
                    self.cleaned_dirs.append(str(full_path))
                    self.total_size_saved += total_size
                    count += 1
                    logger.info(f"删除大型数据目录: {full_path} (大小: {total_size / 1024 / 1024:.2f} MB)")
                except Exception as e:
                    logger.warning(f"删除大型数据目录失败 {full_path}: {e}")
        
        logger.info(f"清理了 {count} 个大型数据目录")
        return count
    
    def clean_report_files(self) -> int:
        """清理报告文件"""
        logger.info("清理报告文件...")
        count = 0
        
        report_patterns = [
            "*_report*.md",
            "*_report*.json",
            "*_analysis*.py",
            "*_analysis*.md",
            "*_validation*.md",
            "validation_plots"
        ]
        
        for pattern in report_patterns:
            for report_file in self.project_root.rglob(pattern):
                if report_file.is_file() or report_file.is_dir():
                    try:
                        if report_file.is_file():
                            size = report_file.stat().st_size
                            report_file.unlink()
                            self.cleaned_files.append(str(report_file))
                            self.total_size_saved += size
                        else:
                            shutil.rmtree(report_file)
                            self.cleaned_dirs.append(str(report_file))
                        count += 1
                        logger.info(f"删除报告文件: {report_file}")
                    except Exception as e:
                        logger.warning(f"删除报告文件失败 {report_file}: {e}")
        
        logger.info(f"清理了 {count} 个报告文件")
        return count
    
    def clean_duplicate_files(self) -> int:
        """清理重复文件"""
        logger.info("清理重复文件...")
        count = 0
        
        # 清理重复的版本化文件
        duplicate_patterns = [
            "*_v*.py",
            "*_v*.json",
            "*_v*.md"
        ]
        
        for pattern in duplicate_patterns:
            for dup_file in self.project_root.rglob(pattern):
                if dup_file.is_file():
                    try:
                        size = dup_file.stat().st_size
                        dup_file.unlink()
                        self.cleaned_files.append(str(dup_file))
                        self.total_size_saved += size
                        count += 1
                        logger.info(f"删除重复文件: {dup_file}")
                    except Exception as e:
                        logger.warning(f"删除重复文件失败 {dup_file}: {e}")
        
        logger.info(f"清理了 {count} 个重复文件")
        return count
    
    def generate_cleanup_report(self):
        """生成清理报告"""
        report_file = self.project_root / "cleanup_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 工程清理报告\n\n")
            
            f.write(f"## 清理统计\n\n")
            f.write(f"- **清理的文件数量**: {len(self.cleaned_files)}\n")
            f.write(f"- **清理的目录数量**: {len(self.cleaned_dirs)}\n")
            f.write(f"- **节省的磁盘空间**: {self.total_size_saved / 1024 / 1024:.2f} MB\n\n")
            
            f.write("## 清理的文件\n\n")
            for file_path in self.cleaned_files[:50]:  # 只显示前50个
                f.write(f"- {file_path}\n")
            
            if len(self.cleaned_files) > 50:
                f.write(f"- ... 还有 {len(self.cleaned_files) - 50} 个文件\n")
            
            f.write("\n## 清理的目录\n\n")
            for dir_path in self.cleaned_dirs:
                f.write(f"- {dir_path}\n")
        
        logger.info(f"清理报告已生成: {report_file}")
    
    def run_full_cleanup(self):
        """运行完整清理"""
        logger.info("开始工程清理...")
        
        total_cleaned = 0
        total_cleaned += self.clean_python_cache()
        total_cleaned += self.clean_log_files()
        total_cleaned += self.clean_test_files()
        total_cleaned += self.clean_backup_files()
        total_cleaned += self.clean_temp_files()
        total_cleaned += self.clean_invalid_results()
        total_cleaned += self.clean_large_data_dirs()
        total_cleaned += self.clean_report_files()
        total_cleaned += self.clean_duplicate_files()
        
        logger.info("=" * 50)
        logger.info("清理完成!")
        logger.info(f"总共清理了 {total_cleaned} 个文件/目录")
        logger.info(f"节省了 {self.total_size_saved / 1024 / 1024:.2f} MB 磁盘空间")
        logger.info("=" * 50)
        
        # 生成清理报告
        self.generate_cleanup_report()

def main():
    """主函数"""
    cleaner = ProjectCleaner()
    cleaner.run_full_cleanup()

if __name__ == "__main__":
    main() 