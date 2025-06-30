# RL-DRAG实验报告

## 实验信息
- 实验名称: rl_drag_comparison
- 工作目录: data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1
- 开始时间: 2025-06-30T18:17:32.447722
- RL训练episodes: 30

## 实验总结

- 实验持续时间: 10672.11秒
- OpenROAD可用: True
- DRAG可用: True
- RL可用: True


## RL训练分析

- 状态: 成功
- 最佳episode: 30
- 最佳奖励: -2445.10
- 最终HPWL: 1.00e+06
- 训练质量: poor
- 收敛episode: 30


## DRAG推荐分析
- 状态: 失败
- 错误: 'DynamicRAGRetriever' object has no attribute 'retrieve_similar_cases'

## 对比分析

- 状态: 成功
- HPWL获胜者: unknown
- 时间获胜者: unknown
- 整体获胜者: DRAG


## 建议
- RL训练效果不佳，建议调整奖励函数或网络结构
- DRAG方法在本次实验中表现更优，建议在实际应用中优先考虑
