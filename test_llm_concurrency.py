#!/usr/bin/env python3
"""
测试LLM并发控制的效果
"""
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from modules.utils.llm_manager import LLMManager
from modules.utils.config_loader import ConfigLoader
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConcurrencyTester:
    def __init__(self, concurrent_limit=2):
        """初始化并发测试器"""
        self.concurrent_limit = concurrent_limit
        self.semaphore = Semaphore(concurrent_limit)
        
        # 初始化LLM管理器
        try:
            config_loader = ConfigLoader()
            llm_config = config_loader.load_config("llm/ollama.json")
            self.llm_manager = LLMManager(llm_config)
        except Exception as e:
            logger.error(f"加载LLM配置失败: {e}")
            self.llm_manager = None
    
    def test_single_request(self, request_id: int) -> dict:
        """测试单个LLM请求"""
        start_time = time.time()
        
        # 获取信号量
        self.semaphore.acquire()
        acquire_time = time.time()
        
        try:
            prompt = f"""
请回答以下问题（请求ID: {request_id}）：

什么是芯片布局设计中的利用率？请简要解释。

请返回JSON格式：
{{
    "request_id": {request_id},
    "answer": "你的答案"
}}
"""
            
            logger.info(f"请求{request_id}: 开始LLM调用")
            model_type = self.llm_manager.select_optimal_model('explanation')
            response = self.llm_manager.generate(prompt, model_type)
            
            end_time = time.time()
            
            result = {
                'request_id': request_id,
                'success': True,
                'response': response,
                'timings': {
                    'total_time': end_time - start_time,
                    'wait_time': acquire_time - start_time,
                    'process_time': end_time - acquire_time
                }
            }
            
            logger.info(f"请求{request_id}: 完成 (总耗时: {result['timings']['total_time']:.2f}s, 等待: {result['timings']['wait_time']:.2f}s, 处理: {result['timings']['process_time']:.2f}s)")
            return result
            
        except Exception as e:
            end_time = time.time()
            result = {
                'request_id': request_id,
                'success': False,
                'error': str(e),
                'timings': {
                    'total_time': end_time - start_time,
                    'wait_time': acquire_time - start_time,
                    'process_time': end_time - acquire_time
                }
            }
            logger.error(f"请求{request_id}: 失败 - {e}")
            return result
        finally:
            self.semaphore.release()
    
    def test_concurrent_requests(self, num_requests: int = 5) -> dict:
        """测试并发请求"""
        if not self.llm_manager:
            logger.error("LLM管理器未初始化")
            return {'error': 'LLM管理器未初始化'}
        
        logger.info(f"开始并发测试: {num_requests}个请求, 并发限制: {self.concurrent_limit}")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            # 提交所有请求
            futures = [executor.submit(self.test_single_request, i) for i in range(num_requests)]
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"并发请求异常: {e}")
                    results.append({'error': str(e)})
        
        end_time = time.time()
        
        # 分析结果
        successful_requests = [r for r in results if r.get('success', False)]
        failed_requests = [r for r in results if not r.get('success', False)]
        
        if successful_requests:
            avg_total_time = sum(r['timings']['total_time'] for r in successful_requests) / len(successful_requests)
            avg_wait_time = sum(r['timings']['wait_time'] for r in successful_requests) / len(successful_requests)
            avg_process_time = sum(r['timings']['process_time'] for r in successful_requests) / len(successful_requests)
        else:
            avg_total_time = avg_wait_time = avg_process_time = 0
        
        summary = {
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'concurrent_limit': self.concurrent_limit,
            'total_test_time': end_time - start_time,
            'average_timings': {
                'total_time': avg_total_time,
                'wait_time': avg_wait_time,
                'process_time': avg_process_time
            },
            'detailed_results': results
        }
        
        return summary

def main():
    """主函数"""
    logger.info("=== LLM并发控制测试 ===")
    
    # 测试不同的并发限制
    test_configs = [
        {'concurrent_limit': 1, 'num_requests': 3},
        {'concurrent_limit': 2, 'num_requests': 4},
        {'concurrent_limit': 3, 'num_requests': 5}
    ]
    
    all_results = {}
    
    for config in test_configs:
        limit = config['concurrent_limit']
        num_req = config['num_requests']
        
        logger.info(f"\n--- 测试配置: 并发限制={limit}, 请求数={num_req} ---")
        
        tester = ConcurrencyTester(concurrent_limit=limit)
        result = tester.test_concurrent_requests(num_requests=num_req)
        all_results[f'limit_{limit}_requests_{num_req}'] = result
        
        # 输出结果摘要
        if 'error' not in result:
            logger.info(f"测试结果:")
            logger.info(f"  总请求数: {result['total_requests']}")
            logger.info(f"  成功请求: {result['successful_requests']}")
            logger.info(f"  失败请求: {result['failed_requests']}")
            logger.info(f"  总测试时间: {result['total_test_time']:.2f}s")
            logger.info(f"  平均总耗时: {result['average_timings']['total_time']:.2f}s")
            logger.info(f"  平均等待时间: {result['average_timings']['wait_time']:.2f}s")
            logger.info(f"  平均处理时间: {result['average_timings']['process_time']:.2f}s")
        else:
            logger.error(f"测试失败: {result['error']}")
    
    # 保存详细结果
    with open('llm_concurrency_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n详细结果已保存到: llm_concurrency_test_results.json")
    
    return all_results

if __name__ == "__main__":
    main() 