import os
import logging
from pathlib import Path
import time
from fetch_ispd_layouts import ISPDLayoutFetcher, ChipDatasetFetcher, DREAMPlaceFetcher

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_images(directory: Path) -> int:
    """统计目录中的图片文件数量"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    return len([f for f in directory.glob('*') if f.suffix.lower() in image_extensions])

def main():
    # 创建输出目录
    output_dir = Path("chiprag/examples/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建PDF目录
    pdf_dir = Path("data/papers/ispd2015")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 从ISPD 2015抓取布局图
    logger.info("开始从ISPD 2015抓取布局图...")
    ispd_fetcher = ISPDLayoutFetcher()
    ispd_fetcher.fetch_from_web()
    ispd_fetcher.fetch_from_pdf("data/papers/ispd2015")
    logger.info("ISPD 2015布局图抓取完成！")
    
    # 等待5秒，避免请求过快
    time.sleep(5)
    
    # 2. 从CHIP数据集抓取布局图
    logger.info("开始从CHIP数据集抓取布局图...")
    chip_fetcher = ChipDatasetFetcher()
    chip_fetcher.fetch_layouts()
    logger.info("CHIP数据集布局图抓取完成！")
    
    # 等待5秒
    time.sleep(5)
    
    # 3. 从DREAMPlace抓取布局图
    logger.info("开始从DREAMPlace抓取布局图...")
    dreamplace_fetcher = DREAMPlaceFetcher()
    dreamplace_fetcher.fetch_layouts()
    logger.info("DREAMPlace布局图抓取完成！")
    
    # 统计结果
    ispd_count = count_images(Path("chiprag/examples/data/ispd_layouts"))
    chip_count = count_images(Path("chiprag/examples/data/chip_dataset"))
    dreamplace_count = count_images(Path("chiprag/examples/data/dreamplace"))
    
    logger.info(f"抓取完成！")
    logger.info(f"ISPD 2015布局图数量: {ispd_count}")
    logger.info(f"CHIP数据集布局图数量: {chip_count}")
    logger.info(f"DREAMPlace布局图数量: {dreamplace_count}")
    logger.info(f"总布局图数量: {ispd_count + chip_count + dreamplace_count}")

if __name__ == "__main__":
    main() 