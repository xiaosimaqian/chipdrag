
# 真实布局测试脚本
puts "开始真实布局测试..."

# 读取设计文件
read_lef tech.lef
read_lef cells.lef
read_def floorplan.def
read_verilog design.v

# 链接设计
link_design des_perf
puts "✅ 设计加载完成"

# 获取设计信息
set design_name [current_design]
puts "设计名称: $design_name"

set cell_count [llength [get_cells]]
puts "单元数量: $cell_count"

set net_count [llength [get_nets]]
puts "网络数量: $net_count"

# 执行布局
puts "开始执行布局..."
global_placement -density 0.91 -init_density_penalty 0.01 -skip_initial_place
puts "✅ 全局布局完成"

detailed_placement
puts "✅ 详细布局完成"

# 检查布局结果
check_placement -verbose
puts "✅ 布局检查完成"

# 获取布局指标
set final_hpwl [get_placement_wirelength]
set final_overflow [get_placement_overflow]
puts "最终HPWL: $final_hpwl"
puts "最终Overflow: $final_overflow"

# 保存布局结果
write_def final_placement.def
puts "✅ 布局结果已保存到 final_placement.def"

# 生成报告
report_placement_wirelength
report_placement_overflow

puts "真实布局测试完成"
