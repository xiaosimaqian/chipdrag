import os
import json
import torch
import matplotlib.pyplot as plt
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.retrieval.dynamic_rag_retriever import DynamicRAGRetriever
from modules.core.layout_generator import LayoutGenerator
from modules.core.rl_agent import QLearningAgent, State

# ========== 路径配置 ==========
ISPD_DIR = "data/designs/ispd_2015_contest_benchmark"
BENCHMARKS = [d for d in os.listdir(ISPD_DIR) if os.path.isdir(os.path.join(ISPD_DIR, d))]

# ========== 配置 ==========
kb_config = {
    "path": "data/knowledge_base/ispd_cases.json",
    "format": "json",
    "layout_experience": "data/knowledge_base"
}
retriever_config = {
    'knowledge_base': kb_config,
    'llm': {'model_name': 'gpt-3.5-turbo', 'temperature': 0.7, 'max_tokens': 1000},
    'dynamic_k_range': (3, 15),
    'quality_threshold': 0.7,
    'learning_rate': 0.1,
    'entity_compression_ratio': 0.1,
    'entity_similarity_threshold': 0.8,
    'compressed_entity_dim': 128
}
rl_config = {
    "alpha": 0.01,
    "gamma": 0.95,
    "epsilon": 0.9,
    "k_range": (3, 15)
}

# ========== DEF文件简单解析与写入 ==========
def parse_def(def_path):
    with open(def_path) as f:
        lines = f.readlines()
    return lines

def write_def(lines, output_path):
    with open(output_path, "w") as f:
        f.writelines(lines)

def update_placement(lines, new_placement):
    # 简单实现：替换PLACEMENT段（假设placement为[{name, x, y, w, h}, ...]）
    # 实际可用pydef等库完善
    in_place = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("- ") and in_place:
            continue  # 跳过原有placement
        if line.strip().startswith("PLACEMENT"):
            in_place = True
            new_lines.append(line)
            for comp in new_placement:
                # DEF格式：- name + PLACED (x y) N ;
                new_lines.append(f"  - {comp['name']} + PLACED ({int(comp['x'])} {int(comp['y'])}) N ;\n")
            continue
        if in_place and line.strip().startswith("END PLACEMENT"):
            in_place = False
            new_lines.append(line)
            continue
        if not in_place:
            new_lines.append(line)
    return new_lines

def visualize_placement(placement, out_path, title="Layout Placement"):
    xs = [comp['x'] for comp in placement]
    ys = [comp['y'] for comp in placement]
    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

# ========== 步骤1：分析原始布局 ==========
def analyze_design(def_path):
    return {"problem": "HPWL高，拥塞热点", "design_name": os.path.basename(os.path.dirname(def_path))}

# ========== 步骤2：构建优化查询 ==========
def build_query(bench, features, problems):
    return {
        "text": f"优化 {bench} 的布局，目标降低HPWL，特征: {features}, 问题: {problems}",
        "design_type": "unknown",
        "complexity": "medium",
        "constraints": {}
    }

# ========== 步骤3：检索知识库+RL agent预测 ==========
def retrieve_knowledge_and_predict(query, design_info):
    knowledge_base = KnowledgeBase(kb_config)
    retriever = DynamicRAGRetriever(retriever_config)
    rag_results = retriever.retrieve_with_dynamic_reranking(query=query, design_info=design_info)
    rl_agent = QLearningAgent(rl_config)
    state = State(
        query_complexity=1.0,
        design_type=design_info.get("design_type", "unknown"),
        constraint_count=len(design_info.get("constraints", {})),
        initial_relevance=1.0,
        result_diversity=1.0,
        historical_performance=0.5,
        timestamp="now"
    )
    action = rl_agent.choose_action(state)
    print(f"RL agent预测最优k值: {action.k_value}")
    return rag_results, action.k_value

# ========== 步骤4：知识增强布局生成 ==========
def generate_new_layout_with_knowledge(rag_results, k_value):
    layout_gen_config = {
        "layout_config": {
            "num_grid_cells": 20 + k_value,
        }
    }
    layout_generator = LayoutGenerator(layout_gen_config)
    features = torch.randn(1, 256)
    layout_result = layout_generator.generate(features)
    return layout_result

# ========== 步骤5：保存新DEF文件 ==========
def save_layout_to_def(layout_result, original_def, output_path):
    lines = parse_def(original_def)
    placement = layout_result.get("placement", [])
    new_lines = update_placement(lines, placement)
    write_def(new_lines, output_path)

# ========== 主控流程 ==========
def main():
    for bench in BENCHMARKS:
        bench_dir = os.path.join(ISPD_DIR, bench)
        original_def = os.path.join(bench_dir, f"{bench}_place.def")
        out_dir = os.path.join(bench_dir, "optimized_outputs")
        os.makedirs(out_dir, exist_ok=True)
        analysis_path = os.path.join(out_dir, "analysis.json")
        if not os.path.exists(original_def):
            print(f"跳过: {bench}，未找到DEF文件: {original_def}")
            continue
        print(f"\n=== 分析: {bench} ===")
        features = extract_def_features(original_def)
        problems = analyze_problems(features)
        query = build_query(bench, features, problems)
        analysis = {
            "design_name": bench,
            "features": features,
            "problems": problems,
            "optimization_query": query
        }
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"分析结果已保存: {analysis_path}")
        print(json.dumps(analysis, indent=2, ensure_ascii=False))

# 简单DEF特征提取
def extract_def_features(def_path):
    features = {
        "num_components": 0,
        "num_pins": 0,
        "num_nets": 0,
        "core_area": 0
    }
    try:
        with open(def_path) as f:
            content = f.read()
        import re
        comp_match = re.search(r'COMPONENTS\s+(\d+)', content)
        if comp_match:
            features["num_components"] = int(comp_match.group(1))
        pins_match = re.search(r'PINS\s+(\d+)', content)
        if pins_match:
            features["num_pins"] = int(pins_match.group(1))
        nets_match = re.search(r'^NETS\s+(\d+)', content, re.MULTILINE)
        if nets_match:
            features["num_nets"] = int(nets_match.group(1))
        diearea_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
        if diearea_match:
            x1, y1, x2, y2 = map(int, diearea_match.groups())
            features["core_area"] = (x2 - x1) * (y2 - y1)
    except Exception as e:
        features["error"] = str(e)
    return features

# 问题点分析（可扩展）
def analyze_problems(features):
    problems = []
    if features["num_components"] > 100000:
        problems.append("组件数极大，布局拥塞风险高")
    if features["core_area"] > 0 and features["num_components"] > 0:
        density = features["num_components"] / features["core_area"]
        if density > 0.01:
            problems.append("核心区密度较高，易拥塞")
    if features["num_nets"] > 50000:
        problems.append("网络数量多，布线复杂度高")
    if not problems:
        problems.append("无明显大问题，常规优化")
    return problems

if __name__ == "__main__":
    main() 