# Test OpenROAD Commands
read_lef tech.lef
read_lef cells.lef
read_def floorplan.def
read_verilog design.v

# Initialize design
link_design des_perf

# Test different report commands
puts "Testing report commands..."

# Test wire length reporting
puts "=== Wire Length Report ==="
report_wire_length -net *

# Test placement overflow
puts "=== Placement Overflow Report ==="
report_placement_overflow

# Test timing
puts "=== Timing Report ==="
report_timing

# Test area
puts "=== Area Report ==="
report_area

# Test power
puts "=== Power Report ==="
report_power

puts "Command test completed." 