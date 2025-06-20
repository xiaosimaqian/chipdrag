import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutImageProcessor:
    """布局图处理器"""
    
    def __init__(self, input_dirs: list, output_dir: str = "chiprag/examples/data/processed_layouts"):
        """
        初始化处理器
        
        Args:
            input_dirs: 输入目录列表
            output_dir: 输出目录
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process_all(self):
        """处理所有布局图"""
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                logger.warning(f"输入目录不存在: {input_dir}")
                continue
                
            logger.info(f"正在处理目录: {input_dir}")
            
            # 获取所有图片文件
            image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
            
            for image_file in tqdm(image_files, desc="处理图片"):
                try:
                    # 处理图片
                    processed_image = self._process_image(image_file)
                    
                    if processed_image is not None:
                        # 保存处理后的图片
                        output_path = self.output_dir / f"processed_{image_file.name}"
                        processed_image.save(output_path)
                        logger.info(f"已保存处理后的图片: {output_path}")
                        
                except Exception as e:
                    logger.error(f"处理图片失败 {image_file}: {str(e)}")
                    
    def _process_image(self, image_path: Path) -> Image.Image:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            处理后的图片
        """
        # 读取图片
        image = Image.open(image_path)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 1. 调整大小
        target_size = (224, 224)  # 模型输入大小
        img_array = cv2.resize(img_array, target_size)
        
        # 2. 图像增强
        # 2.1 对比度增强
        img_array = self._enhance_contrast(img_array)
        
        # 2.2 边缘增强
        img_array = self._enhance_edges(img_array)
        
        # 2.3 噪声去除
        img_array = self._remove_noise(img_array)
        
        # 3. 颜色空间转换
        if len(img_array.shape) == 3:  # 彩色图像
            # 转换为灰度图
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
        # 4. 二值化
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 转回PIL图像
        return Image.fromarray(img_array)
        
    def _enhance_contrast(self, img_array: np.ndarray) -> np.ndarray:
        """
        增强对比度
        
        Args:
            img_array: 图像数组
            
        Returns:
            增强后的图像数组
        """
        if len(img_array.shape) == 3:  # 彩色图像
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # 对L通道进行CLAHE处理
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 合并通道
            lab = cv2.merge((l, a, b))
            
            # 转回RGB
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:  # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(img_array)
            
    def _enhance_edges(self, img_array: np.ndarray) -> np.ndarray:
        """
        增强边缘
        
        Args:
            img_array: 图像数组
            
        Returns:
            增强后的图像数组
        """
        # 使用Canny边缘检测
        edges = cv2.Canny(img_array, 100, 200)
        
        # 膨胀操作
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘叠加到原图
        if len(img_array.shape) == 3:  # 彩色图像
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
        return cv2.addWeighted(img_array, 0.7, edges, 0.3, 0)
        
    def _remove_noise(self, img_array: np.ndarray) -> np.ndarray:
        """
        去除噪声
        
        Args:
            img_array: 图像数组
            
        Returns:
            去噪后的图像数组
        """
        # 使用中值滤波
        return cv2.medianBlur(img_array, 3)

def main():
    """主函数"""
    # 输入目录
    input_dirs = [
        "chiprag/examples/data/ispd_layouts",
        "chiprag/examples/data/online_layouts"
    ]
    
    # 创建处理器
    processor = LayoutImageProcessor(input_dirs)
    
    # 处理所有图片
    processor.process_all()
    
    logger.info("布局图处理完成！")

if __name__ == "__main__":
    main() 