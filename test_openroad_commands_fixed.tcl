# 测试OpenROAD命令的简单脚本
puts "开始测试OpenROAD命令..."

# 读取LEF文件
read_lef tech.lef
read_lef cells.lef
puts "LEF文件读取完成"

# 读取Verilog文件
read_verilog design.v
puts "Verilog文件读取完成"

# 读取DEF文件
read_def floorplan.def
puts "DEF文件读取完成"

# 连接设计
link_design des_perf
puts "设计连接完成"

# 尝试不同的取消布局命令
puts "尝试取消布局..."

# 方法1: 使用set_placement_padding来重置布局
puts "方法1: 使用set_placement_padding"
set_placement_padding -global -left 0 -right 0

# 方法2: 使用global_placement来重新布局
puts "方法2: 使用global_placement重新布局"
global_placement -density 0.7

puts "测试完成"
exit 