import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Ellipse
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

colors = {
    'pretrain': '#FADBD8',
    'finetune': '#D5F5E3',
    'data': '#FDFDFD',
    'model': '#D6EAF8',
    'arrow': '#34495E',
    'arrow_shadow': '#B2BABB',
    'highlight': '#F9CB40'
}

# 阶段色块
ax.add_patch(FancyBboxPatch((1, 8.5), 7, 1.8, boxstyle="round,pad=0.18", facecolor=colors['pretrain'], edgecolor='#222', linewidth=3, alpha=0.97))
ax.text(4.5, 9.4, 'Pre-training Stage', ha='center', va='center', fontsize=18, fontweight='bold', color='#222')
ax.add_patch(FancyBboxPatch((10, 8.5), 7, 1.8, boxstyle="round,pad=0.18", facecolor=colors['finetune'], edgecolor='#222', linewidth=3, alpha=0.97))
ax.text(13.5, 9.4, 'Instruction Fine-tuning Stage', ha='center', va='center', fontsize=18, fontweight='bold', color='#222')

# 子模块均匀分布
pretrain_modules = ["Masked Language Modeling (MLM)", "Contrastive Learning", "Design Element Prediction", "Large-scale Chip Design Data"]
for i, name in enumerate(pretrain_modules):
    x = 2.5 + i*1.7
    y = 8.0
    box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, boxstyle="round,pad=0.15", facecolor=colors['data'], edgecolor='#555', linewidth=2.2)
    ax.add_patch(box)
    ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold', color='#222')
finetune_modules = ["Triplet Data Training", "Adversarial Training", "Knowledge Distillation", "High-quality Labeled Data"]
for i, name in enumerate(finetune_modules):
    x = 11.5 + i*1.7
    y = 8.0
    box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, boxstyle="round,pad=0.15", facecolor=colors['data'], edgecolor='#555', linewidth=2.2)
    ax.add_patch(box)
    ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold', color='#222')

# 模型演进
model_stages = [
    {'name': 'Base Model', 'x': 4.5, 'y': 6.2, 'color': colors['model']},
    {'name': 'Domain Pre-trained Model', 'x': 4.5, 'y': 4.5, 'color': colors['pretrain']},
    {'name': 'Task Fine-tuned Model', 'x': 13.5, 'y': 6.2, 'color': colors['finetune']},
    {'name': 'Final Optimized Model', 'x': 13.5, 'y': 4.5, 'color': colors['highlight']}
]
for stage in model_stages:
    box = FancyBboxPatch((stage['x']-1.1, stage['y']-0.5), 2.2, 1, boxstyle="round,pad=0.13", facecolor=stage['color'], edgecolor='#222', linewidth=2.5)
    ax.add_patch(box)
    ax.text(stage['x'], stage['y'], stage['name'], ha='center', va='center', fontsize=14, fontweight='bold', color='#222')

# 输入输出节点
io_nodes = [
    {'name': 'Technical Docs', 'x': 2.5, 'y': 2.2},
    {'name': 'Design Specs', 'x': 4.5, 'y': 2.2},
    {'name': 'Code Comments', 'x': 6.5, 'y': 2.2},
    {'name': 'Layout Solutions', 'x': 11.5, 'y': 2.2},
    {'name': 'Evaluation Results', 'x': 13.5, 'y': 2.2},
    {'name': 'Expert Feedback', 'x': 15.5, 'y': 2.2}
]
for node in io_nodes:
    ax.add_patch(Ellipse((node['x'], node['y']), 1.5, 0.8, facecolor=colors['data'], edgecolor='#222', linewidth=3))
    ax.text(node['x'], node['y'], node['name'], ha='center', va='center', fontsize=16, fontweight='bold', color='#222')

# 箭头样式
arrow_style = dict(arrowstyle='-|>', mutation_scale=40, fc=colors['arrow'], ec=colors['arrow'], linewidth=4)
shadow_style = dict(arrowstyle='-|>', mutation_scale=40, fc=colors['arrow_shadow'], ec=colors['arrow_shadow'], linewidth=9, alpha=0.22)
# 阶段到模型（主流程加阴影）
for (x, y1, y2) in [(4.5, 8.5, 7.2), (13.5, 8.5, 7.2)]:
    ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **shadow_style))
    ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **arrow_style))
# 模型竖直
for (x, y1, y2) in [(4.5, 6.7, 5.0), (13.5, 6.7, 5.0)]:
    ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **shadow_style))
    ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **arrow_style))
# 横向主流程
for (y, ) in [(6.2,), (4.5,)]:
    ax.add_patch(ConnectionPatch((4.5, y), (13.5, y), "data", "data", **shadow_style))
    ax.add_patch(ConnectionPatch((4.5, y), (13.5, y), "data", "data", **arrow_style))
# 数据流虚线箭头
for node in io_nodes[:3]:
    ax.add_patch(ConnectionPatch((node['x'], 2.6), (node['x'], 3.7), "data", "data", arrowstyle='-|>', mutation_scale=30, fc=colors['arrow'], ec=colors['arrow'], linewidth=2.2, linestyle='dashed'))
for node in io_nodes[3:]:
    ax.add_patch(ConnectionPatch((node['x'], 2.6), (node['x'], 3.7), "data", "data", arrowstyle='-|>', mutation_scale=30, fc=colors['arrow'], ec=colors['arrow'], linewidth=2.2, linestyle='dashed'))

# 技术特点底部两列
features_left = ['• Self-supervised learning', '• Large-scale unlabeled data', '• Domain knowledge injection']
features_right = ['• Supervised learning', '• High-quality labeled data', '• Task adaptation']
for i, feat in enumerate(features_left):
    ax.text(1, 1.1-i*0.3, feat, fontsize=9, color='gray', ha='left', va='top')
for i, feat in enumerate(features_right):
    ax.text(10, 1.1-i*0.3, feat, fontsize=9, color='gray', ha='left', va='top')

# 主标题
ax.text(9, 11.2, 'Two-stage Training Strategy', ha='center', va='center', fontsize=30, fontweight='bold', color='#111')

plt.tight_layout()
plt.savefig('training_strategy.png', dpi=300, bbox_inches='tight')
plt.close()
