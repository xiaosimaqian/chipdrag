# 多模态融合数据需求分析报告

## 1. 当前可用数据评估

### 1.1 文本模态数据 ✅ 充足
- **训练数据**: `data/training/` 包含2000个样本
  - `designs.json` (1.5MB) - 设计描述和约束
  - `queries.json` (591KB) - 布局查询
  - `quality_feedbacks.json` (897KB) - 质量反馈
- **真实数据**: `data/real/` 包含真实项目数据
- **知识库**: `data/knowledge_base/` 包含设计文档

### 1.2 结构化数据 ✅ 充足
- **ISPD基准**: `data/designs/ispd_2015_contest_benchmark/` 包含16个真实电路
  - 每个电路包含: `.v`(网表), `.lef`(库文件), `.def`(布局), `.tcl`(脚本)
- **训练结构化数据**: 包含设计约束、性能指标等JSON格式数据

### 1.3 图像模态数据 ❌ 严重不足
- **当前图像**: `data/test_images/test_layout.png` (286B) - 仅测试用
- **缺失**: 布局图像、布线图像、拥塞热力图等

## 2. 需要补充的图像数据

### 2.1 布局图像数据
```python
# 需要的图像类型和格式
layout_images = {
    "placement_visualization": {
        "format": "PNG/SVG",
        "size": "1920x1080",
        "content": "单元布局位置、连接关系",
        "source": "OpenROAD DEF文件生成"
    },
    "congestion_heatmap": {
        "format": "PNG",
        "size": "1024x1024", 
        "content": "拥塞程度热力图",
        "source": "OpenROAD拥塞分析"
    },
    "timing_critical_path": {
        "format": "PNG",
        "size": "1024x1024",
        "content": "关键路径可视化",
        "source": "OpenROAD时序分析"
    },
    "power_density_map": {
        "format": "PNG", 
        "size": "1024x1024",
        "content": "功耗密度分布图",
        "source": "OpenROAD功耗分析"
    }
}
```

### 2.2 自动化图像生成方法
```python
# 图像生成流程
def generate_layout_images(def_file, output_dir):
    """
    基于DEF文件自动生成布局图像
    """
    # 1. 布局可视化
    tcl_script = f"""
    read_def {def_file}
    gui_start
    gui_show_placement
    gui_save_image {output_dir}/placement.png
    gui_stop
    """
    
    # 2. 拥塞分析
    tcl_script += f"""
    read_def {def_file}
    global_route
    gui_start
    gui_show_congestion
    gui_save_image {output_dir}/congestion.png
    gui_stop
    """
    
    # 3. 时序分析
    tcl_script += f"""
    read_def {def_file}
    read_sdc timing.sdc
    report_timing
    gui_start
    gui_show_timing
    gui_save_image {output_dir}/timing.png
    gui_stop
    """
    
    return run_openroad_with_docker(work_dir, tcl_script)
```

## 3. 多模态融合数据格式

### 3.1 统一数据格式
```json
{
  "design_id": "mgc_des_perf_1",
  "modalities": {
    "text": {
      "design_description": "DES加密模块设计",
      "constraints": ["timing: 1ns", "power: 1W"],
      "queries": ["Generate layout for DES module"]
    },
    "structured": {
      "netlist": "design.v",
      "library": "cells.lef",
      "floorplan": "floorplan.def",
      "constraints": {
        "timing": {"max_delay": "1ns"},
        "power": {"max_power": "1W"},
        "area": {"max_area": "100x100"}
      }
    },
    "images": {
      "placement": "placement.png",
      "congestion": "congestion.png", 
      "timing": "timing.png",
      "power": "power.png"
    }
  },
  "quality_metrics": {
    "wirelength": 12345.67,
    "congestion": 0.85,
    "timing_slack": 0.1,
    "power": 0.95
  }
}
```

### 3.2 数据收集自动化脚本
```python
# 自动化数据收集流程
def collect_multimodal_data(benchmark_dir):
    """
    为每个ISPD基准电路收集多模态数据
    """
    results = {}
    
    for design in os.listdir(benchmark_dir):
        design_path = os.path.join(benchmark_dir, design)
        
        # 1. 收集结构化数据
        structured_data = collect_structured_data(design_path)
        
        # 2. 生成布局图像
        image_data = generate_layout_images(design_path)
        
        # 3. 提取文本描述
        text_data = extract_text_description(design_path)
        
        # 4. 运行布局获取质量指标
        quality_metrics = run_layout_and_evaluate(design_path)
        
        results[design] = {
            "modalities": {
                "text": text_data,
                "structured": structured_data, 
                "images": image_data
            },
            "quality_metrics": quality_metrics
        }
    
    return results
```

## 4. 实施建议

### 4.1 短期目标 (1-2周)
1. **图像生成**: 为现有ISPD基准电路生成布局图像
2. **数据整合**: 将文本、结构化、图像数据整合为统一格式
3. **验证**: 验证多模态数据的质量和完整性

### 4.2 中期目标 (2-4周)  
1. **扩展数据**: 为所有16个ISPD基准电路生成完整多模态数据
2. **质量优化**: 优化图像生成质量，确保可视化效果
3. **自动化**: 完善自动化数据收集流程

### 4.3 长期目标 (1-2月)
1. **数据扩充**: 收集更多类型的芯片设计数据
2. **标注完善**: 为图像数据添加详细标注
3. **验证体系**: 建立多模态数据质量验证体系 