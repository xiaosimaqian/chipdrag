import os
import logging
from datetime import datetime

class TestLogger:
    def __init__(self, test_name):
        # 创建logs目录
        self.log_dir = 'tests/logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'{test_name}_{timestamp}.log')
        
        # 配置日志记录器
        self.logger = logging.getLogger(test_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_data(self, data_name, data):
        """记录数据信息"""
        self.logger.debug(f"处理数据: {data_name}")
        self.logger.debug(f"数据内容: {data}")
    
    def log_result(self, operation, result):
        """记录操作结果"""
        self.logger.info(f"操作: {operation}")
        self.logger.info(f"结果: {result}")
    
    def log_error(self, operation, error):
        """记录错误信息"""
        self.logger.error(f"操作: {operation}")
        self.logger.error(f"错误: {str(error)}")
        self.logger.error(f"错误类型: {type(error)}")
    
    def log_step(self, step_name):
        """记录测试步骤"""
        self.logger.info(f"开始步骤: {step_name}")
    
    def get_log_file(self):
        """获取日志文件路径"""
        return self.log_file 