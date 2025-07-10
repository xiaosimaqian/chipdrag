#!/usr/bin/env python3
"""
OpenROAD输出调试脚本
详细捕获和解析OpenROAD的HPWL输出
"""

import os
import sys
import subprocess
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenROADOutputDebugger:
    """OpenROAD输出调试器"""
    
    def __init__(self, design_dir: str):
        self.design_dir = Path(design_dir)
        self.def_file = None
        self.lef_files = []
        
    def find_design_files(self) -> bool:
        """查找设计文件"""
        def_files = list(self.design_dir.glob("*.def"))
        if not def_files:
            logger.error(f"未找到DEF文件: {self.design_dir}")
            return False
        
        # 优先选择placed.def
        for def_file in def_files:
            if def_file.name == "placed.def":
                self.def_file = def_file
                break
        else:
            self.def_file = def_files[0]
        
        self.lef_files = list(self.design_dir.glob("*.lef"))
        if not self.lef_files:
            logger.error(f"未找到LEF文件: {self.design_dir}")
            return False
        
        logger.info(f"DEF文件: {self.def_file}")
        logger.info(f"LEF文件: {[f.name for f in self.lef_files]}")
        return True
    
    def create_detailed_tcl_script(self) -> str:
        """创建详细的TCL调试脚本"""
        tcl_script = """# OpenROAD详细输出调试脚本
puts "=== OpenROAD详细输出调试开始 ==="

# 重置数据库
if {[info exists ::ord::db]} {
    ord::reset_db
}

# 读取LEF文件
"""
        
        for lef_file in self.lef_files:
            tcl_script += f"read_lef {lef_file.name}\n"
        
        tcl_script += f"""
# 读取DEF文件
read_def {self.def_file.name}

puts "文件读取完成"

# 获取设计信息
set db [ord::get_db]
set chip [$db getChip]
set block [$chip getBlock]
set nets [$block getNets]

puts "设计信息："
puts "  网络数: [llength $nets]"

# 统计网络类型
set signal_nets 0
set power_nets 0
set clock_nets 0

foreach net $nets {{
    set sig_type [$net getSigType]
    if {{$sig_type == "SIGNAL"}} {{
        incr signal_nets
    }} elseif {{$sig_type == "POWER"}} {{
        incr power_nets
    }} elseif {{$sig_type == "CLOCK"}} {{
        incr clock_nets
    }}
}}

puts "信号网络数: $signal_nets"
puts "电源网络数: $power_nets"
puts "时钟网络数: $clock_nets"

# 详细测试report_wire_length输出
puts "=== 开始详细测试report_wire_length输出 ==="

set test_nets 0
set max_test_nets 5

foreach net $nets {{
    if {{[$net getSigType] == "SIGNAL"}} {{
        set net_name [$net getName]
        puts "\\n=== 测试网络: $net_name ==="
        
        # 方法1: 直接调用report_wire_length
        puts "方法1: 直接调用report_wire_length"
        if {{[catch {{
            report_wire_length -net $net_name
        }} err]}} {{
            puts "方法1失败: $err"
        }}
        
        # 方法2: 尝试捕获返回值
        puts "方法2: 尝试捕获返回值"
        if {{[catch {{
            set result [report_wire_length -net $net_name]
            puts "返回值: $result"
        }} err]}} {{
            puts "方法2失败: $err"
        }}
        
        # 方法3: 使用exec命令
        puts "方法3: 使用exec命令"
        if {{[catch {{
            set result [exec report_wire_length -net $net_name]
            puts "exec结果: $result"
        }} err]}} {{
            puts "方法3失败: $err"
        }}
        
        # 方法4: 检查网络属性
        puts "方法4: 检查网络属性"
        puts "网络名称: [$net getName]"
        puts "网络类型: [$net getSigType]"
        puts "网络状态: [$net isSpecial]"
        
        # 方法5: 尝试获取网络长度属性
        puts "方法5: 尝试获取网络长度属性"
        if {{[catch {{
            set length [$net getLength]
            puts "网络长度: $length"
        }} err]}} {{
            puts "方法5失败: $err"
        }}
        
        # 方法6: 尝试获取网络边界框
        puts "方法6: 尝试获取网络边界框"
        if {{[catch {{
            set bbox [$net getBBox]
            if {{$bbox != "NULL"}} {{
                puts "边界框: [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax]"
            }} else {{
                puts "边界框: NULL"
            }}
        }} err]}} {{
            puts "方法6失败: $err"
        }}
        
        incr test_nets
        if {{$test_nets >= $max_test_nets}} {{
            puts "已达到测试网络数量限制: $max_test_nets"
            break
        }}
    }}
}}

puts "\\n=== 测试其他可能的HPWL命令 ==="

# 测试其他命令
set test_commands {{
    "report_design_area"
    "report_checks"
    "report_utilization"
    "report_net_stats"
    "report_clock_utilization"
}}

foreach cmd $test_commands {{
    puts "\\n测试命令: $cmd"
    if {{[catch {{
        $cmd
    }} err]}} {{
        puts "命令 $cmd 失败: $err"
    }} else {{
        puts "命令 $cmd 成功"
    }}
}}

puts "\\n=== OpenROAD详细输出调试完成 ==="
exit
"""
        
        return tcl_script
    
    def debug_openroad_output(self) -> Dict:
        """调试OpenROAD输出"""
        if not self.find_design_files():
            return {"error": "未找到设计文件"}
        
        # 创建TCL脚本
        tcl_script = self.create_detailed_tcl_script()
        tcl_file = self.design_dir / "debug_output.tcl"
        
        with open(tcl_file, 'w') as f:
            f.write(tcl_script)
        
        logger.info(f"TCL脚本已写入: {tcl_file}")
        
        # 执行OpenROAD命令
        logger.info("执行OpenROAD命令...")
        
        try:
            result = subprocess.run([
                'openroad', '-exit', 'debug_output.tcl'
            ], capture_output=True, text=True, timeout=120, cwd=self.design_dir)
            
            logger.info(f"OpenROAD执行完成，返回码: {result.returncode}")
            
            # 详细分析输出
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n') if result.stderr else []
            
            # 解析输出
            debug_info = {
                "design_dir": str(self.design_dir),
                "def_file": str(self.def_file),
                "lef_files": [str(f) for f in self.lef_files],
                "return_code": result.returncode,
                "output_lines": output_lines,
                "error_lines": error_lines,
                "analysis": self._analyze_output(output_lines, error_lines)
            }
            
            return debug_info
            
        except subprocess.TimeoutExpired:
            return {"error": "OpenROAD执行超时"}
        except Exception as e:
            return {"error": f"OpenROAD执行异常: {e}"}
    
    def _analyze_output(self, output_lines: List[str], error_lines: List[str]) -> Dict:
        """分析OpenROAD输出"""
        analysis = {
            "hpwl_patterns": [],
            "successful_commands": [],
            "failed_commands": [],
            "network_info": {},
            "potential_hpwl_values": []
        }
        
        # 查找HPWL相关模式
        hpwl_patterns = [
            r'wire length:\s*([\d.]+)',
            r'length:\s*([\d.]+)',
            r'HPWL:\s*([\d.]+)',
            r'wirelength:\s*([\d.]+)',
            r'([\d.]+)\s*um',
            r'([\d.]+)\s*units'
        ]
        
        for line in output_lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查HPWL模式
            for pattern in hpwl_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    analysis["hpwl_patterns"].append({
                        "line": line,
                        "pattern": pattern,
                        "value": value
                    })
                    analysis["potential_hpwl_values"].append(value)
            
            # 检查成功/失败的命令
            if "成功" in line or "success" in line.lower():
                analysis["successful_commands"].append(line)
            elif "失败" in line or "error" in line.lower() or "invalid" in line.lower():
                analysis["failed_commands"].append(line)
            
            # 检查网络信息
            if "网络:" in line or "net:" in line.lower():
                analysis["network_info"][line] = True
        
        return analysis

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python debug_openroad_output.py <design_dir>")
        sys.exit(1)
    
    design_dir = sys.argv[1]
    debugger = OpenROADOutputDebugger(design_dir)
    
    results = debugger.debug_openroad_output()
    
    # 输出结果
    print("\n=== OpenROAD输出调试结果 ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # 保存结果到文件
    output_file = Path(design_dir) / "openroad_debug_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 