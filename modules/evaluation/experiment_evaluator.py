import numpy as np

def check_constraints(layout, constraints):
    total_area = sum(comp['width'] * comp['height'] for comp in layout['components'])
    return total_area <= constraints['max_area']

def calc_timing_margin(layout):
    return float(np.random.uniform(0.7, 1.0))

def calc_area(layout):
    return sum(comp['width'] * comp['height'] for comp in layout['components'])

def calc_power(layout):
    return float(np.random.uniform(2.5, 3.5))

def calc_knowledge_reuse(layout, knowledge_base):
    reused = 0
    for comp in layout['components']:
        if comp['id'] in knowledge_base:
            reused += 1
    return reused / len(layout['components']) if layout['components'] else 0

def evaluate_layout(layout, constraints, knowledge_base):
    feasible = check_constraints(layout, constraints)
    timing_margin = calc_timing_margin(layout)
    area = calc_area(layout)
    power = calc_power(layout)
    score = 0.4 * timing_margin + 0.3 * (1 - area/constraints['max_area']) + 0.3 * (1 - power/constraints['max_power'])
    reuse_rate = calc_knowledge_reuse(layout, knowledge_base)
    return {
        'feasible': feasible,
        'timing_margin': round(timing_margin, 3),
        'area': round(area, 2),
        'power': round(power, 2),
        'score': round(score, 3),
        'reuse_rate': round(reuse_rate, 3)
    } 