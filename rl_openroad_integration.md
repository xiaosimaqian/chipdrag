# RL与OpenROAD联动设计文档

## 1. RL系统架构设计

### 1.1 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RL Agent      │    │   OpenROAD      │    │   Environment   │
│                 │    │   Interface     │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ State       │ │    │ │ TCL Script  │ │    │ │ Layout      │ │
│ │ Processing  │ │◄──►│ │ Generator   │ │◄──►│ │ Evaluation  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Action      │ │    │ │ Docker      │ │    │ │ Quality     │ │
│ │ Selection   │ │    │ │ Executor    │ │    │ │ Metrics     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Reward      │ │    │ │ Result      │ │    │ │ Feedback    │ │
│ │ Function    │ │    │ │ Parser      │ │    │ │ Generator   │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 1.2 RL输入定义

#### 状态空间 (State Space)
```python
class LayoutState:
    def __init__(self):
        self.design_features = {
            "cell_count": int,           # 单元数量
            "net_count": int,            # 网络数量
            "design_area": float,        # 设计面积
            "constraint_count": int,     # 约束数量
            "design_type": str,          # 设计类型 (DSP, Memory, etc.)
            "complexity_score": float    # 复杂度评分
        }
        
        self.retrieval_features = {
            "current_k": int,            # 当前检索文档数
            "retrieval_history": list,   # 历史检索效果
            "query_complexity": float,   # 查询复杂度
            "knowledge_coverage": float  # 知识覆盖度
        }
        
        self.quality_features = {
            "current_wirelength": float, # 当前线长
            "current_congestion": float, # 当前拥塞
            "current_timing": float,     # 当前时序
            "current_power": float,      # 当前功耗
            "improvement_rate": float    # 改进率
        }
```

#### 动作空间 (Action Space)
```python
class LayoutAction:
    def __init__(self):
        self.retrieval_actions = {
            "k_value": range(3, 16),     # 检索文档数量 [3, 15]
            "retrieval_strategy": [      # 检索策略
                "semantic",              # 语义检索
                "constraint_based",      # 约束检索
                "experience_based",      # 经验检索
                "hybrid"                 # 混合检索
            ]
        }
        
        self.placement_actions = {
            "density_target": [0.7, 0.8, 0.9, 0.95],  # 密度目标
            "placement_algorithm": [     # 布局算法
                "global_placement",
                "detailed_placement", 
                "incremental_placement"
            ],
            "optimization_focus": [      # 优化重点
                "wirelength",
                "congestion", 
                "timing",
                "power",
                "balanced"
            ]
        }
```

### 1.3 RL训练流程

#### 训练算法
```python
class LayoutRLAgent:
    def __init__(self):
        self.q_table = {}  # Q值表
        self.epsilon = 0.9  # 探索率
        self.alpha = 0.01   # 学习率
        self.gamma = 0.95   # 折扣因子
        
    def train_episode(self, design_path):
        """单次训练episode"""
        # 1. 初始化状态
        state = self.get_initial_state(design_path)
        
        # 2. 执行布局优化循环
        for step in range(MAX_STEPS):
            # 选择动作
            action = self.select_action(state)
            
            # 执行动作 (调用OpenROAD)
            next_state, reward = self.execute_action(state, action, design_path)
            
            # 更新Q值
            self.update_q_value(state, action, reward, next_state)
            
            # 状态转移
            state = next_state
            
            # 检查终止条件
            if self.is_terminated(state):
                break
                
        return self.calculate_episode_reward()
    
    def execute_action(self, state, action, design_path):
        """执行动作 - 调用OpenROAD"""
        # 1. 生成TCL脚本
        tcl_script = self.generate_tcl_script(state, action)
        
        # 2. 调用OpenROAD
        result = run_openroad_with_docker(
            work_dir=Path(design_path),
            cmd=tcl_script,
            is_tcl=True
        )
        
        # 3. 解析结果
        quality_metrics = self.parse_openroad_output(result.stdout)
        
        # 4. 计算奖励
        reward = self.calculate_reward(quality_metrics)
        
        # 5. 构建下一状态
        next_state = self.build_next_state(state, action, quality_metrics)
        
        return next_state, reward
```

### 1.4 奖励函数设计

#### 多目标奖励函数
```python
def calculate_reward(self, quality_metrics):
    """计算多目标奖励"""
    # 基础奖励
    wirelength_reward = self.calculate_wirelength_reward(quality_metrics['wirelength'])
    congestion_reward = self.calculate_congestion_reward(quality_metrics['congestion'])
    timing_reward = self.calculate_timing_reward(quality_metrics['timing_slack'])
    power_reward = self.calculate_power_reward(quality_metrics['power'])
    
    # 加权组合
    total_reward = (
        0.3 * wirelength_reward +
        0.25 * congestion_reward +
        0.25 * timing_reward +
        0.2 * power_reward
    )
    
    # 约束违反惩罚
    constraint_penalty = self.calculate_constraint_penalty(quality_metrics)
    
    # 效率奖励 (改进速度)
    efficiency_reward = self.calculate_efficiency_reward(quality_metrics)
    
    return total_reward - constraint_penalty + efficiency_reward

def calculate_wirelength_reward(self, wirelength):
    """线长奖励计算"""
    # 基于历史最优值的相对改进
    baseline = self.get_baseline_wirelength()
    improvement = (baseline - wirelength) / baseline
    return max(0, improvement)  # 非负奖励

def calculate_congestion_reward(self, congestion):
    """拥塞奖励计算"""
    # 拥塞越低越好
    return max(0, 1.0 - congestion)

def calculate_timing_reward(self, timing_slack):
    """时序奖励计算"""
    # 时序裕量越大越好
    return max(0, timing_slack)

def calculate_power_reward(self, power):
    """功耗奖励计算"""
    # 功耗越低越好
    baseline = self.get_baseline_power()
    improvement = (baseline - power) / baseline
    return max(0, improvement)
```

## 2. OpenROAD集成接口

### 2.1 TCL脚本生成器
```python
class TCLScriptGenerator:
    def __init__(self):
        self.template_dir = "templates/"
        
    def generate_placement_script(self, state, action):
        """生成布局TCL脚本"""
        script = f"""
# 动态布局脚本 - 基于RL状态和动作生成
puts "开始动态布局优化..."

# 读取设计文件
read_lef tech.lef
read_lef cells.lef
read_def floorplan.def
read_verilog design.v

# 链接设计
link_design des_perf

# 设置布局参数 (来自RL动作)
set density_target {action['placement_actions']['density_target']}
set optimization_focus "{action['placement_actions']['optimization_focus']}"

# 执行布局
puts "执行{action['placement_actions']['placement_algorithm']}..."
{action['placement_actions']['placement_algorithm']} -density $density_target

# 根据优化重点进行特定优化
if {{$optimization_focus == "wirelength"}} {{
    optimize_wirelength
}} elseif {{$optimization_focus == "congestion"}} {{
    optimize_congestion
}} elseif {{$optimization_focus == "timing"}} {{
    optimize_timing
}} elseif {{$optimization_focus == "power"}} {{
    optimize_power
}}

# 输出质量指标
report_wirelength
report_congestion
report_timing
report_power

# 保存结果
write_def final_placement.def
puts "布局完成"
"""
        return script
    
    def generate_routing_script(self, state, action):
        """生成布线TCL脚本"""
        script = f"""
# 动态布线脚本
puts "开始动态布线优化..."

read_def final_placement.def

# 全局布线
global_route

# 详细布线
detailed_route

# 输出布线质量
report_wirelength
report_congestion
report_drc

write_def final_routed.def
puts "布线完成"
"""
        return script
```

### 2.2 结果解析器
```python
class OpenROADResultParser:
    def parse_placement_output(self, output):
        """解析布局输出"""
        metrics = {}
        
        # 解析线长
        wirelength_match = re.search(r'Total wirelength: ([\d.]+)', output)
        if wirelength_match:
            metrics['wirelength'] = float(wirelength_match.group(1))
            
        # 解析拥塞
        congestion_match = re.search(r'Congestion: ([\d.]+)', output)
        if congestion_match:
            metrics['congestion'] = float(congestion_match.group(1))
            
        # 解析时序
        timing_match = re.search(r'Worst slack: ([\d.-]+)', output)
        if timing_match:
            metrics['timing_slack'] = float(timing_match.group(1))
            
        # 解析功耗
        power_match = re.search(r'Total power: ([\d.]+)', output)
        if power_match:
            metrics['power'] = float(power_match.group(1))
            
        return metrics
    
    def parse_routing_output(self, output):
        """解析布线输出"""
        metrics = {}
        
        # 解析DRC违规
        drc_match = re.search(r'DRC violations: (\d+)', output)
        if drc_match:
            metrics['drc_violations'] = int(drc_match.group(1))
            
        # 解析布线完成率
        route_match = re.search(r'Routed: (\d+)/(\d+)', output)
        if route_match:
            routed = int(route_match.group(1))
            total = int(route_match.group(2))
            metrics['route_completion'] = routed / total
            
        return metrics
```

## 3. RL输出定义

### 3.1 策略输出
```python
class RLPolicyOutput:
    def __init__(self):
        self.retrieval_policy = {
            "optimal_k": int,           # 最优检索文档数
            "retrieval_strategy": str,  # 最优检索策略
            "confidence_score": float   # 策略置信度
        }
        
        self.placement_policy = {
            "optimal_density": float,   # 最优密度目标
            "placement_algorithm": str, # 最优布局算法
            "optimization_focus": str,  # 最优优化重点
            "iteration_count": int      # 建议迭代次数
        }
        
        self.quality_prediction = {
            "predicted_wirelength": float,
            "predicted_congestion": float,
            "predicted_timing": float,
            "predicted_power": float,
            "confidence_interval": tuple
        }
```

### 3.2 训练输出
```python
class RLTrainingOutput:
    def __init__(self):
        self.training_metrics = {
            "episode_rewards": list,    # 每轮奖励
            "convergence_curve": list,  # 收敛曲线
            "policy_improvement": float # 策略改进程度
        }
        
        self.learned_policy = {
            "q_table": dict,           # 学习到的Q值表
            "optimal_actions": dict,   # 最优动作映射
            "state_value_function": dict # 状态价值函数
        }
        
        self.validation_results = {
            "test_performance": dict,  # 测试性能
            "generalization_score": float, # 泛化能力
            "robustness_analysis": dict   # 鲁棒性分析
        }
```

## 4. 实施计划

### 4.1 第一阶段：基础集成 (1周)
1. 实现基础RL智能体
2. 集成OpenROAD调用接口
3. 实现简单奖励函数
4. 在单个设计上验证

### 4.2 第二阶段：优化完善 (2周)
1. 完善多目标奖励函数
2. 实现经验回放机制
3. 优化TCL脚本生成
4. 扩展状态和动作空间

### 4.3 第三阶段：批量训练 (1周)
1. 在所有ISPD基准上训练
2. 实现训练监控和可视化
3. 验证策略泛化能力
4. 生成训练报告

### 4.4 第四阶段：实验验证 (1周)
1. 与基线方法对比
2. 消融实验验证
3. 性能分析和优化
4. 生成最终实验结果 