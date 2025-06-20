import os
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISPDLayoutFetcher:
    def __init__(self):
        self.base_url = "https://ispd.cc/contests/15/web"
        self.output_dir = Path("chiprag/examples/data/ispd_layouts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_from_web(self):
        """从ISPD 2015网站抓取布局图"""
        urls = [
            f"{self.base_url}/benchmark.html",  # 更新为正确的benchmark页面
            f"{self.base_url}/downloads.html"
        ]
        
        for url in urls:
            try:
                logger.info(f"正在抓取网页: {url}")
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找所有图片链接
                for img in soup.find_all('img'):
                    img_url = img.get('src')
                    if img_url:
                        if not img_url.startswith('http'):
                            img_url = f"{self.base_url}/{img_url}"
                        
                        if self._is_layout_image(img_url):
                            self._download_image(img_url)
                            
            except Exception as e:
                logger.error(f"抓取网页失败: {str(e)}")
                
    def fetch_from_pdf(self, pdf_path: str):
        """从PDF文件中提取布局图"""
        try:
            pdf_dir = Path(pdf_path)
            if not pdf_dir.exists():
                logger.warning(f"PDF目录不存在: {pdf_path}")
                return
                
            for pdf_file in pdf_dir.glob("*.pdf"):
                try:
                    doc = fitz.open(pdf_file)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        images = page.get_images()
                        
                        for img_index, img in enumerate(images):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # 转换为PIL Image对象
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            if self._is_layout_image_content(image):
                                # 保存图片
                                output_path = self.output_dir / f"{pdf_file.stem}_page{page_num+1}_img{img_index+1}.png"
                                image.save(output_path)
                                logger.info(f"已保存PDF中的图片: {output_path}")
                                
                except Exception as e:
                    logger.error(f"处理PDF文件失败: {str(e)}")
                    
        except Exception as e:
            logger.error(f"处理PDF文件失败: {str(e)}")
            
    def _is_layout_image(self, url: str) -> bool:
        """判断URL是否指向布局图"""
        layout_keywords = ['layout', 'floorplan', 'placement', 'routing', 'benchmark']
        return any(keyword in url.lower() for keyword in layout_keywords)
        
    def _is_layout_image_content(self, image: Image.Image) -> bool:
        """判断图片内容是否为布局图"""
        # 检查图片尺寸比例
        width, height = image.size
        aspect_ratio = width / height
        
        # 布局图通常具有特定的宽高比
        if 0.5 <= aspect_ratio <= 2.0:
            return True
        return False
        
    def _download_image(self, url: str):
        """下载图片"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 从URL中提取文件名
            filename = url.split('/')[-1]
            output_path = self.output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"已下载图片: {output_path}")
            
        except Exception as e:
            logger.error(f"下载图片失败: {str(e)}")

class ChipDatasetFetcher:
    def __init__(self):
        self.base_url = "https://chip-dataset.vercel.app"
        self.output_dir = Path("chiprag/examples/data/chip_dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_layouts(self):
        """从CHIP数据集抓取布局图"""
        try:
            # 由于CHIP数据集需要认证，我们直接使用已知的图片URL
            image_urls = [
                "https://chip-dataset.vercel.app/images/cpu-die-size.png",
                "https://chip-dataset.vercel.app/images/gpu-die-size.png",
                "https://chip-dataset.vercel.app/images/cpu-transistors.png",
                "https://chip-dataset.vercel.app/images/gpu-transistors.png",
                "https://chip-dataset.vercel.app/images/cpu-frequency.png",
                "https://chip-dataset.vercel.app/images/gpu-frequency.png"
            ]
            
            for url in image_urls:
                self._download_image(url)
                    
        except Exception as e:
            logger.error(f"从CHIP数据集抓取失败: {str(e)}")
            
    def _is_layout_image(self, url: str) -> bool:
        """判断URL是否指向布局图"""
        layout_keywords = ['layout', 'chip', 'die', 'floorplan']
        return any(keyword in url.lower() for keyword in layout_keywords)
        
    def _download_image(self, url: str):
        """下载图片"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = url.split('/')[-1]
            output_path = self.output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"已下载CHIP数据集图片: {output_path}")
            
        except Exception as e:
            logger.error(f"下载CHIP数据集图片失败: {str(e)}")

class DREAMPlaceFetcher:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/limbo018/DREAMPlace/master"
        self.output_dir = Path("chiprag/examples/data/dreamplace")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_layouts(self):
        """从DREAMPlace仓库抓取布局图"""
        try:
            # 使用GitHub raw content URL
            image_urls = [
                f"{self.base_url}/docs/figures/ispd2005_adaptec1.png",
                f"{self.base_url}/docs/figures/ispd2005_adaptec2.png",
                f"{self.base_url}/docs/figures/ispd2005_adaptec3.png",
                f"{self.base_url}/docs/figures/ispd2005_adaptec4.png",
                f"{self.base_url}/docs/figures/ispd2005_bigblue1.png",
                f"{self.base_url}/docs/figures/ispd2005_bigblue2.png",
                f"{self.base_url}/docs/figures/ispd2005_bigblue3.png",
                f"{self.base_url}/docs/figures/ispd2005_bigblue4.png"
            ]
            
            for url in image_urls:
                self._download_image(url)
                    
        except Exception as e:
            logger.error(f"从DREAMPlace抓取失败: {str(e)}")
            
    def _is_layout_image(self, url: str) -> bool:
        """判断URL是否指向布局图"""
        layout_keywords = ['layout', 'placement', 'routing', 'floorplan']
        return any(keyword in url.lower() for keyword in layout_keywords)
        
    def _download_image(self, url: str):
        """下载图片"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            filename = url.split('/')[-1]
            output_path = self.output_dir / filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"已下载DREAMPlace图片: {output_path}")
            
        except Exception as e:
            logger.error(f"下载DREAMPlace图片失败: {str(e)}")

def main():
    # 创建必要的目录
    Path("chiprag/examples/data").mkdir(parents=True, exist_ok=True)
    Path("data/papers/ispd2015").mkdir(parents=True, exist_ok=True)
    
    # 抓取ISPD 2015布局图
    ispd_fetcher = ISPDLayoutFetcher()
    ispd_fetcher.fetch_from_web()
    ispd_fetcher.fetch_from_pdf("data/papers/ispd2015")
    
    # 抓取CHIP数据集布局图
    chip_fetcher = ChipDatasetFetcher()
    chip_fetcher.fetch_layouts()
    
    # 抓取DREAMPlace布局图
    dreamplace_fetcher = DREAMPlaceFetcher()
    dreamplace_fetcher.fetch_layouts()

if __name__ == "__main__":
    main() 