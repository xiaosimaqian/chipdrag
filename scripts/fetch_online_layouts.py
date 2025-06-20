import os
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
import time
from urllib.parse import urljoin
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OnlineLayoutFetcher:
    """在线布局图抓取器"""
    
    def __init__(self, output_dir: str = "chiprag/examples/data/online_layouts"):
        """
        初始化抓取器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 无头模式
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # 初始化WebDriver
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # 布局图相关网站
        self.layout_sources = [
            {
                'name': 'OpenCores',
                'url': 'https://opencores.org/',
                'type': 'project'
            },
            {
                'name': 'GitHub',
                'url': 'https://github.com/search?q=vlsi+layout',
                'type': 'code'
            },
            {
                'name': 'ResearchGate',
                'url': 'https://www.researchgate.net/search/publication?q=vlsi%20layout',
                'type': 'paper'
            }
        ]
        
    def fetch_all(self):
        """抓取所有来源的布局图"""
        for source in self.layout_sources:
            try:
                logger.info(f"正在抓取 {source['name']} 的布局图")
                
                if source['type'] == 'project':
                    self._fetch_from_project_site(source['url'])
                elif source['type'] == 'code':
                    self._fetch_from_code_site(source['url'])
                elif source['type'] == 'paper':
                    self._fetch_from_paper_site(source['url'])
                    
                time.sleep(2)  # 避免请求过快
                
            except Exception as e:
                logger.error(f"抓取 {source['name']} 失败: {str(e)}")
                
    def _fetch_from_project_site(self, url: str):
        """
        从项目网站抓取布局图
        
        Args:
            url: 网站URL
        """
        try:
            self.driver.get(url)
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
            
            # 查找所有图片
            images = self.driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                img_url = img.get_attribute('src')
                if img_url and self._is_layout_image(img_url):
                    self._download_image(img_url, 'project')
                    
        except Exception as e:
            logger.error(f"从项目网站抓取失败: {str(e)}")
            
    def _fetch_from_code_site(self, url: str):
        """
        从代码网站抓取布局图
        
        Args:
            url: 网站URL
        """
        try:
            self.driver.get(url)
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
            
            # 查找所有图片
            images = self.driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                img_url = img.get_attribute('src')
                if img_url and self._is_layout_image(img_url):
                    self._download_image(img_url, 'code')
                    
        except Exception as e:
            logger.error(f"从代码网站抓取失败: {str(e)}")
            
    def _fetch_from_paper_site(self, url: str):
        """
        从论文网站抓取布局图
        
        Args:
            url: 网站URL
        """
        try:
            self.driver.get(url)
            
            # 等待页面加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
            
            # 查找所有图片
            images = self.driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                img_url = img.get_attribute('src')
                if img_url and self._is_layout_image(img_url):
                    self._download_image(img_url, 'paper')
                    
        except Exception as e:
            logger.error(f"从论文网站抓取失败: {str(e)}")
            
    def _is_layout_image(self, url: str) -> bool:
        """
        判断URL是否指向布局图
        
        Args:
            url: 图片URL
            
        Returns:
            是否是布局图
        """
        # 检查URL中是否包含布局相关关键词
        layout_keywords = [
            'layout', 'floorplan', 'placement', 'routing',
            'vlsi', 'ic', 'chip', 'circuit', 'design'
        ]
        return any(keyword in url.lower() for keyword in layout_keywords)
        
    def _download_image(self, url: str, source_type: str):
        """
        下载图片
        
        Args:
            url: 图片URL
            source_type: 来源类型
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 从URL中提取文件名
            filename = url.split('/')[-1]
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += '.png'
                
            # 添加来源类型前缀
            filename = f"{source_type}_{filename}"
            
            # 保存图片
            output_path = self.output_dir / filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"已下载图片: {output_path}")
            
        except Exception as e:
            logger.error(f"下载图片失败: {str(e)}")
            
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    """主函数"""
    # 创建抓取器
    fetcher = OnlineLayoutFetcher()
    
    # 抓取所有来源的布局图
    fetcher.fetch_all()
    
    logger.info("布局图抓取完成！")

if __name__ == "__main__":
    main() 