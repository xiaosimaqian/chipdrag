# chiprag/modules/utils/pdf_processor.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
import pdfplumber
from tqdm import tqdm
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF文件处理器"""
    
    def __init__(self, base_dir: str):
        """初始化PDF处理器
        
        Args:
            base_dir: 知识库基础目录
        """
        self.base_dir = Path(base_dir)
        self.text_dir = self.base_dir / "text"
        self._init_directories()
        
    def _init_directories(self):
        """初始化目录结构"""
        # 创建主目录
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        subdirs = [
            "design_docs",      # 设计文档
            "technical_specs",  # 技术规范
            "research_papers",  # 研究论文
            "tutorials"        # 教程文档
        ]
        
        for subdir in subdirs:
            (self.text_dir / subdir).mkdir(exist_ok=True)
            
    def process_pdf(self, pdf_path: str, category: str) -> Dict:
        """处理单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            category: 文档类别
            
        Returns:
            Dict: 处理结果
        """
        try:
            # 检查文件是否存在
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
                
            # 检查类别是否有效
            if category not in ["design_docs", "technical_specs", "research_papers", "tutorials"]:
                raise ValueError(f"无效的文档类别: {category}")
                
            # 创建目标目录
            target_dir = self.text_dir / category
            target_dir.mkdir(exist_ok=True)
            
            # 提取文本
            text_content = self._extract_text(pdf_path)
            
            # 生成元数据
            metadata = self._generate_metadata(pdf_path, category)
            
            # 保存处理结果
            result = {
                "content": text_content,
                "metadata": metadata
            }
            
            # 保存到JSON文件
            output_path = target_dir / f"{pdf_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            return result
            
        except Exception as e:
            logger.error(f"处理PDF文件时出错: {str(e)}")
            raise
            
    def _extract_text(self, pdf_path: Path) -> str:
        """提取PDF文件中的文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            str: 提取的文本
        """
        try:
            # 使用pdfplumber提取文本
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in tqdm(pdf.pages, desc="提取文本"):
                    text += page.extract_text() + "\n"
                return text
                
        except Exception as e:
            logger.error(f"提取文本时出错: {str(e)}")
            raise
            
    def _generate_metadata(self, pdf_path: Path, category: str) -> Dict:
        """生成PDF文件的元数据
        
        Args:
            pdf_path: PDF文件路径
            category: 文档类别
            
        Returns:
            Dict: 元数据
        """
        try:
            # 使用PyPDF2获取PDF信息
            with open(pdf_path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                info = pdf.metadata
                
            return {
                "filename": pdf_path.name,
                "category": category,
                "title": info.get("/Title", ""),
                "author": info.get("/Author", ""),
                "creation_date": info.get("/CreationDate", ""),
                "page_count": len(pdf.pages),
                "file_size": pdf_path.stat().st_size,
                "processing_date": str(datetime.now())
            }
            
        except Exception as e:
            logger.error(f"生成元数据时出错: {str(e)}")
            raise
            
    def process_directory(self, dir_path: str, category: str):
        """处理目录中的所有PDF文件
        
        Args:
            dir_path: 目录路径
            category: 文档类别
        """
        try:
            dir_path = Path(dir_path)
            if not dir_path.exists():
                raise FileNotFoundError(f"目录不存在: {dir_path}")
                
            # 处理所有PDF文件
            for pdf_file in tqdm(list(dir_path.glob("**/*.pdf")), desc="处理PDF文件"):
                try:
                    self.process_pdf(str(pdf_file), category)
                except Exception as e:
                    logger.error(f"处理文件 {pdf_file} 时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"处理目录时出错: {str(e)}")
            raise
            
    def create_index(self) -> Dict:
        """创建文本索引
        
        Returns:
            Dict: 索引数据
        """
        try:
            index = {}
            
            # 遍历所有JSON文件
            for json_file in self.text_dir.glob("**/*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                    # 添加到索引
                    index[json_file.stem] = {
                        "category": data["metadata"]["category"],
                        "title": data["metadata"]["title"],
                        "path": str(json_file.relative_to(self.text_dir))
                    }
                    
                except Exception as e:
                    logger.error(f"处理索引文件 {json_file} 时出错: {str(e)}")
                    continue
                    
            # 保存索引
            index_path = self.text_dir / "index.json"
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
                
            return index
            
        except Exception as e:
            logger.error(f"创建索引时出错: {str(e)}")
            raise

class PDFProcessingStatus:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.status_file = self.cache_dir / 'pdf_processing_status.json'
        self.status = self._load_status()
        
    def _load_status(self) -> Dict:
        """加载处理状态"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_status(self):
        """保存处理状态"""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
            
    def is_processed(self, pdf_path: str) -> bool:
        """检查PDF是否已处理"""
        pdf_hash = self._get_file_hash(pdf_path)
        return pdf_hash in self.status
        
    def mark_as_processed(self, pdf_path: str, metadata: Dict):
        """标记PDF为已处理"""
        pdf_hash = self._get_file_hash(pdf_path)
        self.status[pdf_hash] = {
            'path': pdf_path,
            'processed_time': datetime.now().isoformat(),
            'metadata': metadata
        }
        self._save_status()
        
    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()