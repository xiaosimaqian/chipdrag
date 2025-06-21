#!/usr/bin/env python3
"""
数据收集示例
展示如何收集芯片设计领域的真实数据
"""

import json
import logging
import requests
import time
from pathlib import Path
from datetime import datetime
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_github_data():
    """从GitHub收集开源设计项目数据"""
    logger.info("开始从GitHub收集数据...")
    
    # 搜索关键词
    search_terms = ['RISC-V', 'DSP', 'ASIC', 'FPGA']
    collected_data = []
    
    for term in search_terms:
        try:
            # 搜索仓库
            url = "https://api.github.com/search/repositories"
            params = {
                'q': f'{term} language:verilog language:vhdl',
                'sort': 'stars',
                'order': 'desc',
                'per_page': 5
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                repos = data.get('items', [])
                
                for repo in repos:
                    repo_info = {
                        'name': repo['full_name'],
                        'description': repo.get('description', ''),
                        'language': repo.get('language', ''),
                        'stars': repo.get('stargazers_count', 0),
                        'topics': repo.get('topics', []),
                        'url': repo.get('html_url', ''),
                        'search_term': term,
                        'timestamp': datetime.now().isoformat()
                    }
                    collected_data.append(repo_info)
                
                logger.info(f"找到 {len(repos)} 个 {term} 相关仓库")
            
            # 避免API限制
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"搜索 {term} 失败: {str(e)}")
    
    return collected_data

def generate_sample_queries(num_queries=50):
    """生成示例查询数据"""
    logger.info(f"生成 {num_queries} 个示例查询...")
    
    queries = []
    design_types = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
    constraint_types = ['timing', 'power', 'area', 'reliability', 'yield']
    
    query_templates = [
        "Generate layout for {design_type} with {constraints} constraints",
        "Design {design_type} layout optimized for {constraints}",
        "Create {design_type} layout with focus on {constraints}",
        "Optimize {design_type} layout for {constraints} requirements"
    ]
    
    for i in range(num_queries):
        design_type = random.choice(design_types)
        num_constraints = random.randint(1, 3)
        constraints = random.sample(constraint_types, num_constraints)
        
        template = random.choice(query_templates)
        query_text = template.format(
            design_type=design_type,
            constraints=', '.join(constraints)
        )
        
        query = {
            'query_id': f'query_{i:04d}',
            'query_text': query_text,
            'design_type': design_type,
            'constraints': constraints,
            'complexity': random.uniform(0.3, 0.9),
            'user_id': f'user_{random.randint(1, 10):02d}',
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic'
        }
        
        queries.append(query)
    
    return queries

def generate_sample_designs(num_designs=30):
    """生成示例设计数据"""
    logger.info(f"生成 {num_designs} 个示例设计...")
    
    designs = []
    design_types = ['risc_v', 'dsp', 'memory', 'accelerator', 'controller']
    tech_nodes = ['14nm', '28nm', '40nm', '65nm']
    
    for i in range(num_designs):
        design_type = random.choice(design_types)
        tech_node = random.choice(tech_nodes)
        
        design = {
            'design_id': f'design_{i:04d}',
            'design_type': design_type,
            'technology_node': tech_node,
            'area_constraint': random.uniform(1000, 10000),
            'power_budget': random.uniform(1.0, 10.0),
            'timing_constraint': random.uniform(1.0, 5.0),
            'constraints': random.sample(['timing', 'power', 'area', 'reliability'], random.randint(1, 3)),
            'netlist_file': f'{design_type}_{i:04d}.v',
            'def_file': f'{design_type}_{i:04d}.def',
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic'
        }
        
        designs.append(design)
    
    return designs

def generate_sample_results(num_results=40):
    """生成示例布局结果数据"""
    logger.info(f"生成 {num_results} 个示例布局结果...")
    
    results = []
    
    for i in range(num_results):
        result = {
            'result_id': f'result_{i:04d}',
            'query_id': f'query_{random.randint(0, 49):04d}',
            'design_id': f'design_{random.randint(0, 29):04d}',
            'layout_file': f'layout_{i:04d}.def',
            'wirelength': random.uniform(5000, 50000),
            'congestion': random.uniform(0.05, 0.25),
            'timing_score': random.uniform(0.6, 0.95),
            'power_score': random.uniform(0.6, 0.95),
            'area_utilization': random.uniform(0.7, 0.95),
            'generation_time': random.uniform(10, 300),
            'method': random.choice(['chip_d_rag', 'traditional', 'baseline']),
            'timestamp': datetime.now().isoformat()
        }
        
        results.append(result)
    
    return results

def generate_sample_feedback(num_feedback=30):
    """生成示例质量反馈数据"""
    logger.info(f"生成 {num_feedback} 个示例质量反馈...")
    
    feedbacks = []
    
    for i in range(num_feedback):
        feedback = {
            'feedback_id': f'feedback_{i:04d}',
            'result_id': f'result_{random.randint(0, 39):04d}',
            'user_id': f'user_{random.randint(1, 10):02d}',
            'wirelength_score': random.uniform(0.5, 1.0),
            'congestion_score': random.uniform(0.5, 1.0),
            'timing_score': random.uniform(0.5, 1.0),
            'power_score': random.uniform(0.5, 1.0),
            'overall_score': random.uniform(0.6, 0.95),
            'feedback_text': random.choice([
                "Good layout quality, meets timing requirements",
                "Layout is well optimized for power",
                "Area utilization is efficient",
                "Some congestion issues in certain areas",
                "Timing closure achieved successfully"
            ]),
            'timestamp': datetime.now().isoformat()
        }
        
        feedbacks.append(feedback)
    
    return feedbacks

def save_data(data, filename):
    """保存数据到文件"""
    output_dir = Path('data/real')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据已保存到 {output_file}")

def generate_statistics(data, data_type):
    """生成数据统计"""
    if not data:
        return
    
    stats = {
        'data_type': data_type,
        'total_count': len(data),
        'collection_timestamp': datetime.now().isoformat()
    }
    
    # 根据数据类型生成特定统计
    if data_type == 'github_repos':
        stats['language_distribution'] = {}
        stats['topic_distribution'] = {}
        
        for item in data:
            if 'language' in item:
                lang = item['language']
                stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
            
            if 'topics' in item:
                for topic in item['topics']:
                    stats['topic_distribution'][topic] = stats['topic_distribution'].get(topic, 0) + 1
    
    elif data_type == 'queries':
        stats['design_type_distribution'] = {}
        stats['constraint_distribution'] = {}
        
        for item in data:
            if 'design_type' in item:
                design_type = item['design_type']
                stats['design_type_distribution'][design_type] = stats['design_type_distribution'].get(design_type, 0) + 1
            
            if 'constraints' in item:
                for constraint in item['constraints']:
                    stats['constraint_distribution'][constraint] = stats['constraint_distribution'].get(constraint, 0) + 1
    
    elif data_type == 'designs':
        stats['design_type_distribution'] = {}
        stats['technology_distribution'] = {}
        
        for item in data:
            if 'design_type' in item:
                design_type = item['design_type']
                stats['design_type_distribution'][design_type] = stats['design_type_distribution'].get(design_type, 0) + 1
            
            if 'technology_node' in item:
                tech_node = item['technology_node']
                stats['technology_distribution'][tech_node] = stats['technology_distribution'].get(tech_node, 0) + 1
    
    elif data_type == 'results':
        stats['method_distribution'] = {}
        
        for item in data:
            if 'method' in item:
                method = item['method']
                stats['method_distribution'][method] = stats['method_distribution'].get(method, 0) + 1
    
    # 保存统计报告
    output_dir = Path('data/real')
    stats_file = output_dir / f'{data_type}_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"统计报告已生成: {stats_file}")

def main():
    """主函数"""
    logger.info("开始数据收集示例...")
    
    # 1. 从GitHub收集数据
    try:
        github_data = collect_github_data()
        if github_data:
            save_data(github_data, 'github_repos.json')
            generate_statistics(github_data, 'github_repos')
    except Exception as e:
        logger.error(f"GitHub数据收集失败: {str(e)}")
    
    # 2. 生成示例查询
    queries = generate_sample_queries(50)
    save_data(queries, 'sample_queries.json')
    generate_statistics(queries, 'queries')
    
    # 3. 生成示例设计
    designs = generate_sample_designs(30)
    save_data(designs, 'sample_designs.json')
    generate_statistics(designs, 'designs')
    
    # 4. 生成示例结果
    results = generate_sample_results(40)
    save_data(results, 'sample_results.json')
    generate_statistics(results, 'results')
    
    # 5. 生成示例反馈
    feedbacks = generate_sample_feedback(30)
    save_data(feedbacks, 'sample_feedback.json')
    generate_statistics(feedbacks, 'feedback')
    
    # 6. 生成综合报告
    summary = {
        'collection_summary': {
            'github_repos': len(github_data) if 'github_data' in locals() else 0,
            'queries': len(queries),
            'designs': len(designs),
            'results': len(results),
            'feedback': len(feedbacks)
        },
        'total_data_points': len(queries) + len(designs) + len(results) + len(feedbacks),
        'collection_timestamp': datetime.now().isoformat(),
        'data_quality': 'synthetic_and_real',
        'usage_note': 'This data can be used for training and testing Chip-D-RAG system'
    }
    
    save_data(summary, 'data_collection_summary.json')
    
    logger.info("数据收集示例完成！")
    logger.info(f"总共收集了 {summary['total_data_points']} 个数据点")

if __name__ == '__main__':
    main() 