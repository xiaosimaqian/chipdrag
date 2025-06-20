import numpy as np
from typing import List, Dict, Any

class LayoutOptimizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_spacing = self.config.get('min_spacing', 0.05)

    # ===================== 目标函数 =====================
    def calc_total_overlap(self, placement: List[List[float]]) -> float:
        n = len(placement)
        overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                overlap += self._overlap_area(placement[i], placement[j])
        return overlap

    def _overlap_area(self, comp1, comp2) -> float:
        x1, y1, w1, h1 = comp1[:4]
        x2, y2, w2, h2 = comp2[:4]
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        return x_overlap * y_overlap

    def calc_total_wirelength(self, placement: List[List[float]]) -> float:
        # 以所有组件中心的完全连通线长为例
        n = len(placement)
        wirelength = 0.0
        for i in range(n):
            for j in range(i+1, n):
                xi, yi = placement[i][0] + placement[i][2]/2, placement[i][1] + placement[i][3]/2
                xj, yj = placement[j][0] + placement[j][2]/2, placement[j][1] + placement[j][3]/2
                wirelength += np.sqrt((xi-xj)**2 + (yi-yj)**2)
        return wirelength

    def check_space_utilization(self, placement: List[List[float]], region_area=4.0, fill_limit=0.7):
        total_area = sum(comp[2]*comp[3] for comp in placement)
        fill_rate = total_area / region_area
        print(f"总组件面积: {total_area:.2f}, 布局区域面积: {region_area}, 填充率: {fill_rate:.2f}")
        if fill_rate > fill_limit:
            print("警告：填充率过高，物理上无法无重叠布局！")
        return fill_rate

    def energy(self, placement: List[List[float]]) -> float:
        overlap = self.calc_total_overlap(placement)
        if overlap > 0:
            return 1e6 + overlap * 1e6  # 极大惩罚
        return self.calc_total_wirelength(placement)

    # ===================== 模拟退火 =====================
    def simulated_annealing_optimize(self, placement: List[List[float]], max_iter=1000, T0=1.0, alpha=0.995) -> List[List[float]]:
        T = T0
        best = [list(comp) for comp in placement]
        best_energy = self.energy(best)
        current = [list(comp) for comp in placement]
        for i in range(max_iter):
            candidate = self._random_perturb(current)
            e1 = self.energy(current)
            e2 = self.energy(candidate)
            if e2 < e1 or np.random.rand() < np.exp((e1 - e2) / T):
                current = candidate
                if e2 < best_energy:
                    best = [list(comp) for comp in candidate]
                    best_energy = e2
            T *= alpha
        return best

    def _random_perturb(self, placement: List[List[float]]) -> List[List[float]]:
        idx = np.random.randint(len(placement))
        new_placement = [list(comp) for comp in placement]
        new_placement[idx][0] += np.random.uniform(-0.05, 0.05)
        new_placement[idx][1] += np.random.uniform(-0.05, 0.05)
        return new_placement

    # ===================== 遗传算法 =====================
    def genetic_optimize(self, placement: List[List[float]], pop_size=30, generations=100, mutation_rate=0.2) -> List[List[float]]:
        def crossover(p1, p2):
            # 单点交叉
            idx = np.random.randint(len(p1))
            child = [list(p1[i]) if i < idx else list(p2[i]) for i in range(len(p1))]
            return child
        def mutate(ind):
            idx = np.random.randint(len(ind))
            ind[idx][0] += np.random.uniform(-0.05, 0.05)
            ind[idx][1] += np.random.uniform(-0.05, 0.05)
            return ind
        # 初始化种群
        population = [[list(comp) for comp in placement] for _ in range(pop_size)]
        for gen in range(generations):
            scores = [self.energy(ind) for ind in population]
            idx = np.argsort(scores)
            population = [population[i] for i in idx]
            # 精英保留
            new_pop = population[:2]
            # 交叉
            while len(new_pop) < pop_size:
                p1, p2 = population[np.random.randint(pop_size//2)], population[np.random.randint(pop_size//2)]
                child = crossover(p1, p2)
                if np.random.rand() < mutation_rate:
                    child = mutate(child)
                new_pop.append(child)
            population = new_pop
        return population[0]

    # ===================== 力导向算法 =====================
    def force_directed_optimize(self, placement: List[List[float]], max_iter=200, k_rep=0.01, k_att=0.01) -> List[List[float]]:
        n = len(placement)
        pos = np.array([[comp[0], comp[1]] for comp in placement])
        for _ in range(max_iter):
            disp = np.zeros_like(pos)
            # 斥力
            for i in range(n):
                for j in range(n):
                    if i != j:
                        delta = pos[i] - pos[j]
                        dist = np.linalg.norm(delta) + 1e-6
                        disp[i] += k_rep * delta / dist**2
            # 引力（以原始网格点为锚点）
            for i in range(n):
                anchor = np.array([placement[i][0], placement[i][1]])
                disp[i] += k_att * (anchor - pos[i])
            pos += disp
        # 更新位置
        new_placement = [list(comp) for comp in placement]
        for i in range(n):
            new_placement[i][0], new_placement[i][1] = pos[i][0], pos[i][1]
        return new_placement

    # ===================== 统一接口 =====================
    def optimize(self, placement: List[List[float]], method="anneal") -> List[List[float]]:
        self.check_space_utilization(placement)
        if method == "anneal":
            return self.simulated_annealing_optimize(placement)
        elif method == "genetic":
            return self.genetic_optimize(placement)
        elif method == "force":
            return self.force_directed_optimize(placement)
        else:
            # 默认多轮重叠检测与调整
            return self._local_overlap_optimize(placement)

    def _local_overlap_optimize(self, placement: List[List[float]], max_iter: int = 100) -> List[List[float]]:
        for _ in range(max_iter):
            changed = False
            n = len(placement)
            for i in range(n):
                for j in range(i+1, n):
                    if self._check_overlap(placement[i], placement[j]):
                        dx = placement[j][0] - placement[i][0]
                        dy = placement[j][1] - placement[i][1]
                        if abs(dx) > abs(dy):
                            placement[j][0] += self.min_spacing * (1 if dx > 0 else -1)
                        else:
                            placement[j][1] += self.min_spacing * (1 if dy > 0 else -1)
                        changed = True
            if not changed:
                for comp in placement:
                    comp[0] += np.random.uniform(-0.01, 0.01)
                    comp[1] += np.random.uniform(-0.01, 0.01)
                break
        return placement

    def _check_overlap(self, comp1, comp2) -> bool:
        x1, y1, w1, h1 = comp1[:4]
        x2, y2, w2, h2 = comp2[:4]
        return not (
            x1 + w1 + self.min_spacing <= x2 or
            x2 + w2 + self.min_spacing <= x1 or
            y1 + h1 + self.min_spacing <= y2 or
            y2 + h2 + self.min_spacing <= y1
        ) 