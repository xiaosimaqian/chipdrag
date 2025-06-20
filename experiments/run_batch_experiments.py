import json, csv, os
import datetime
from modules.evaluation.experiment_evaluator import evaluate_layout

cases = ['alu', 'fifo', 'uart']
results = []
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for case in cases:
    base = f'data/test_cases/{case}'
    # 合成netlist和constraints
    if not os.path.exists(f'{base}/constraints.json'):
        os.makedirs(base, exist_ok=True)
        with open(f'{base}/constraints.json', 'w') as f:
            json.dump({"max_area": 12000, "max_power": 3.5, "max_timing": 1.0}, f)
    constraints = json.load(open(f'{base}/constraints.json'))
    # 合成layout
    layout = {
        'components': [
            {'id': f'C{i}', 'width': 0.08, 'height': 0.08} for i in range(1, 101)
        ]
    }
    # 合成knowledge_base
    kb_path = 'data/knowledge_base/components.json'
    if not os.path.exists(kb_path):
        os.makedirs(os.path.dirname(kb_path), exist_ok=True)
        with open(kb_path, 'w') as f:
            json.dump({f'C{i}': {"type": "GEN", "width": 0.08, "height": 0.08} for i in range(1, 101)}, f)
    knowledge_base = json.load(open(kb_path))
    metrics = evaluate_layout(layout, constraints, knowledge_base)
    metrics['case'] = case
    results.append(metrics)

# 保存CSV
os.makedirs('results', exist_ok=True)
csv_path = f'results/experiment_summary_{timestamp}.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"实验结果已保存到 {csv_path}")

def to_latex_table(results):
    header = ' & '.join(results[0].keys()) + ' \\ \\hline'
    rows = [' & '.join(str(v) for v in r.values()) + ' \\' for r in results]
    return '\\begin{tabular}{|' + 'c|'*len(results[0]) + '}' + '\n\\hline\n' + header + '\n' + '\n'.join(rows) + '\n\\hline\n\\end{tabular}'

tex_path = f'results/experiment_summary_{timestamp}.tex'
with open(tex_path, 'w') as f:
    f.write(to_latex_table(results))
print(f"LaTeX表格已保存到 {tex_path}") 
