import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Ellipse

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

colors = {
    'layer1': '#D6EAF8',
    'layer2': '#D5F5E3',
    'layer3': '#FADBD8',
    'module': '#FDFDFD',
    'arrow': '#34495E',
    'arrow_shadow': '#B2BABB'
}

layer_defs = [
    {'name': 'Design Intent Understanding Layer', 'y': 8.2, 'color': colors['layer1']},
    {'name': 'Constraint Modeling Layer', 'y': 5.7, 'color': colors['layer2']},
    {'name': 'Domain Term Embedding Layer', 'y': 3.2, 'color': colors['layer3']}
]

# 主层色块
for layer in layer_defs:
    main_box = FancyBboxPatch((1, layer['y']-0.8), 10, 1.6, boxstyle="round,pad=0.18", facecolor=layer['color'], edgecolor='#222', linewidth=3, alpha=0.97)
    ax.add_patch(main_box)
    ax.text(6, layer['y'], layer['name'], ha='center', va='center', fontsize=18, fontweight='bold', color='#222')

# 子模块均匀分布
modules = [
    ["Multi-head Attention Mechanism", "Intent Reasoning Module", "Global Objective Integration"],
    ["Graph Neural Network (GNN)", "Message Passing Mechanism", "Hierarchical Structure Learning"],
    ["Term Embedding Module", "Term Normalization Module", "Contextual Relation Learning Module"]
]
for i, layer in enumerate(layer_defs):
    y = layer['y']-1.1
    for j, name in enumerate(modules[i]):
        x = 3 + j*3
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, boxstyle="round,pad=0.15", facecolor=colors['module'], edgecolor='#555', linewidth=2.2)
        ax.add_patch(box)
        ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold', color='#222')

# 输入输出用大椭圆
ax.add_patch(Ellipse((0.8, 3.2), 1.6, 0.9, facecolor='#FDFDFD', edgecolor='#222', linewidth=3))
ax.text(0.8, 3.2, 'Chip Design Terms', ha='center', va='center', fontsize=16, fontweight='bold', color='#222')
ax.add_patch(Ellipse((11.2, 8.2), 1.8, 0.9, facecolor='#FDFDFD', edgecolor='#222', linewidth=3))
ax.text(11.2, 8.2, 'Unified Semantic\nRepresentation', ha='center', va='center', fontsize=16, fontweight='bold', color='#222')

# 注释右下角分两列
tech_notes_left = [
    '• 768-dim vector',
    '• Term mapping',
    '• Normalization',
    '• 1024-dim visual feature'
]
tech_notes_right = [
    '• 512-dim structured feature',
    '• Multi-objective balance',
    '• Intent reasoning'
]
for i, note in enumerate(tech_notes_left):
    ax.text(9.5, 1.2-i*0.25, note, fontsize=9, color='gray', ha='left', va='top')
for i, note in enumerate(tech_notes_right):
    ax.text(10.7, 1.2-i*0.25, note, fontsize=9, color='gray', ha='left', va='top')

# 箭头（主流程加阴影）
arrow_style = dict(arrowstyle='-|>', mutation_scale=40, fc=colors['arrow'], ec=colors['arrow'], linewidth=4)
shadow_style = dict(arrowstyle='-|>', mutation_scale=40, fc=colors['arrow_shadow'], ec=colors['arrow_shadow'], linewidth=8, alpha=0.25)
# 层间主流程
for x in [6]:
    for y1, y2 in [(3.2+0.3, 5.7-0.8), (5.7+0.8, 8.2-0.8)]:
        # 阴影
        ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **shadow_style))
        # 主箭头
        ax.add_patch(ConnectionPatch((x, y1), (x, y2), "data", "data", **arrow_style))
# 输入到embedding
ax.add_patch(ConnectionPatch((1.6, 3.2), (2.3, 3.2), "data", "data", **arrow_style))
# 输出
ax.add_patch(ConnectionPatch((10.7, 8.2), (10.2, 8.2), "data", "data", **arrow_style))

# 主标题
ax.text(6, 9.5, 'Chip Semantic Encoder Architecture', ha='center', va='center', fontsize=30, fontweight='bold', color='#111')

plt.tight_layout()
plt.savefig('chip_semantic_encoder.png', dpi=300, bbox_inches='tight')
plt.close()