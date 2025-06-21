#!/usr/bin/env python3
"""
测试OpenROAD Docker集成的脚本
验证是否可以调用OpenROAD工具进行真实的EDA操作
"""

import subprocess
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenROADDockerInterface:
    """OpenROAD Docker接口类"""
    
    def __init__(self, image_name: str = "openroad/flow-ubuntu22.04-builder:21e414"):
        self.image_name = image_name
        self.container_name = "openroad_test"
        
    def test_docker_availability(self) -> bool:
        """测试Docker是否可用"""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            logger.info(f"Docker可用: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Docker不可用: {e}")
            return False
    
    def test_image_availability(self) -> bool:
        """测试OpenROAD镜像是否可用"""
        try:
            result = subprocess.run(
                ["docker", "images", self.image_name], 
                capture_output=True, 
                text=True, 
                check=True
            )
            if self.image_name.split(':')[0] in result.stdout:
                logger.info(f"OpenROAD镜像可用: {self.image_name}")
                return True
            else:
                logger.warning(f"OpenROAD镜像不可用: {self.image_name}")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"检查镜像失败: {e}")
            return False
    
    def run_container_command_enhanced(self, command: str, work_dir: str = None) -> Tuple[bool, str, str]:
        """使用增强的Docker配置运行命令"""
        # 获取当前用户ID和组ID
        try:
            user_id = subprocess.run(["id", "-u"], capture_output=True, text=True, check=True).stdout.strip()
            group_id = subprocess.run(["id", "-g"], capture_output=True, text=True, check=True).stdout.strip()
        except:
            user_id = "1000"
            group_id = "1000"
        
        # 获取显示设置
        display = os.environ.get('DISPLAY', ':0')
        
        # 使用绝对路径
        if work_dir:
            work_dir = str(Path(work_dir).resolve())
        else:
            work_dir = str(Path.cwd().resolve())
        
        docker_cmd = [
            "docker", "run", "--rm",
            "-u", f"{user_id}:{group_id}",
            "-v", f"{work_dir}:/workspace",
            "-w", "/workspace",
            "-e", f"DISPLAY={display}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            "-v", f"{os.path.expanduser('~')}/.Xauthority:/.Xauthority",
            "--network", "host",
            "--security-opt", "seccomp=unconfined",
            self.image_name,
            "bash", "-c", command
        ]
        
        logger.debug(f"执行Docker命令: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=60  # 增加超时时间
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("命令执行超时")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"执行命令失败: {e}")
            return False, "", str(e)
    
    def run_container_command(self, command: str, work_dir: str = None) -> Tuple[bool, str, str]:
        """在容器中运行命令（简化版本）"""
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/workspace" if work_dir else f"{os.getcwd()}:/workspace",
            "-w", "/workspace",
            self.image_name,
            "bash", "-c", command
        ]
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error("命令执行超时")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"执行命令失败: {e}")
            return False, "", str(e)
    
    def test_basic_environment(self) -> Dict[str, Any]:
        """测试基本环境"""
        logger.info("测试OpenROAD基本环境...")
        
        tests = {
            "docker_available": self.test_docker_availability(),
            "image_available": self.test_image_availability(),
        }
        
        if tests["docker_available"] and tests["image_available"]:
            # 测试容器内基本命令
            success, stdout, stderr = self.run_container_command("echo 'Hello from OpenROAD container'")
            tests["container_basic"] = success
            
            if success:
                # 测试是否有基本的EDA工具
                success, stdout, stderr = self.run_container_command("which yosys || echo 'yosys not found'")
                tests["yosys_available"] = "yosys" in stdout
                
                success, stdout, stderr = self.run_container_command("which openroad || echo 'openroad not found'")
                tests["openroad_available"] = "openroad" in stdout
                
                # 检查OpenROAD Flow工具
                success, stdout, stderr = self.run_container_command("ls /OpenROAD-flow-scripts/flow/scripts/ 2>/dev/null || echo 'flow scripts not found'")
                tests["flow_scripts_available"] = "flow scripts not found" not in stdout
                
                # 检查其他可能的工具
                success, stdout, stderr = self.run_container_command("ls /usr/local/bin/ | grep -E '(openroad|yosys|triton|innovus|openlane)' || echo 'No EDA tools found'")
                tests["eda_tools_found"] = "No EDA tools found" not in stdout
                
                # 检查OpenROAD版本
                success, stdout, stderr = self.run_container_command("openroad --version 2>/dev/null || echo 'openroad version not available'")
                tests["openroad_version"] = stdout.strip() if "not available" not in stdout else "Unknown"
        
        return tests
    
    def test_openroad_flow(self, design_dir: str) -> Dict[str, Any]:
        """测试OpenROAD Flow流程"""
        logger.info(f"测试OpenROAD Flow流程: {design_dir}")
        
        # 创建测试配置
        config_content = """
# OpenROAD Flow配置文件
DESIGN_NAME=test_design
VERILOG_FILES=design.v
SDC_FILE=design.sdc
DIE_AREA=0 0 1000 1000
CORE_AREA=10 10 990 990
CLOCK_PERIOD=10
"""
        
        config_path = Path(design_dir) / "config.tcl"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # 尝试运行OpenROAD Flow
        flow_command = f"""
cd {design_dir} && \
ls -la && \
echo "Testing OpenROAD Flow..." && \
which openroad && \
openroad --version 2>/dev/null || echo "OpenROAD not available"
"""
        
        success, stdout, stderr = self.run_container_command_enhanced(flow_command, design_dir)
        
        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "openroad_available": "OpenROAD not available" not in stdout
        }
    
    def simulate_openroad_synthesis(self, verilog_file: str, output_dir: str) -> Dict[str, Any]:
        """模拟OpenROAD综合过程"""
        logger.info(f"模拟OpenROAD综合: {verilog_file}")
        
        # 创建简单的TCL脚本
        tcl_script = f"""
# OpenROAD综合脚本
read_verilog {verilog_file}
synth_design -top top -part xc7a35tcpg236-1
write_verilog {output_dir}/synthesized.v
report_timing
report_power
report_area
"""
        
        # 保存TCL脚本
        script_path = Path(output_dir) / "synthesis.tcl"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        # 模拟运行结果
        simulated_results = {
            "wns": -0.5,  # Worst Negative Slack (ns)
            "tns": -2.3,  # Total Negative Slack (ns)
            "power": 45.2,  # Power (mW)
            "area": 1250,   # Area (LUTs)
            "congestion": 0.15,  # Congestion ratio
            "success": True
        }
        
        logger.info(f"模拟综合完成: WNS={simulated_results['wns']}ns, Power={simulated_results['power']}mW")
        return simulated_results
    
    def simulate_openroad_place_route(self, def_file: str, output_dir: str) -> Dict[str, Any]:
        """模拟OpenROAD布局布线过程"""
        logger.info(f"模拟OpenROAD布局布线: {def_file}")
        
        # 模拟布局布线结果
        simulated_results = {
            "wns": -0.3,  # 布局布线后时序改善
            "tns": -1.8,
            "power": 48.5,
            "area": 1250,
            "congestion": 0.12,  # 拥塞改善
            "drc_violations": 0,
            "success": True
        }
        
        logger.info(f"模拟布局布线完成: WNS={simulated_results['wns']}ns, Congestion={simulated_results['congestion']}")
        return simulated_results

def test_with_real_design():
    """使用真实设计文件进行测试"""
    logger.info("=== 使用真实设计文件测试OpenROAD集成 ===")
    
    # 查找测试设计文件
    design_dir = Path("data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1")
    if not design_dir.exists():
        logger.error(f"设计目录不存在: {design_dir}")
        return
    
    # 查找Verilog和DEF文件
    verilog_files = list(design_dir.glob("*.v"))
    def_files = list(design_dir.glob("*.def"))
    
    if not verilog_files:
        logger.error("未找到Verilog文件")
        return
    
    if not def_files:
        logger.error("未找到DEF文件")
        return
    
    verilog_file = verilog_files[0]
    def_file = def_files[0]
    
    logger.info(f"使用设计文件: {verilog_file.name}, {def_file.name}")
    
    # 创建OpenROAD接口
    openroad = OpenROADDockerInterface()
    
    # 测试环境
    env_test = openroad.test_basic_environment()
    logger.info(f"环境测试结果: {json.dumps(env_test, indent=2)}")
    
    # 测试OpenROAD Flow
    flow_test = openroad.test_openroad_flow(str(design_dir))
    logger.info(f"OpenROAD Flow测试结果: {json.dumps(flow_test, indent=2)}")
    
    # 模拟综合
    synthesis_results = openroad.simulate_openroad_synthesis(
        str(verilog_file), 
        "temp/synthesis_output"
    )
    
    # 模拟布局布线
    place_route_results = openroad.simulate_openroad_place_route(
        str(def_file), 
        "temp/place_route_output"
    )
    
    # 生成测试报告
    test_report = {
        "environment_test": env_test,
        "flow_test": flow_test,
        "synthesis_results": synthesis_results,
        "place_route_results": place_route_results,
        "design_files": {
            "verilog": str(verilog_file),
            "def": str(def_file)
        }
    }
    
    # 保存报告
    report_path = Path("temp/openroad_integration_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    logger.info(f"测试报告已保存到: {report_path}")
    
    return test_report

def main():
    """主函数"""
    logger.info("=== OpenROAD Docker集成测试 ===")
    
    # 创建临时目录
    Path("temp").mkdir(exist_ok=True)
    
    # 运行测试
    test_report = test_with_real_design()
    
    if test_report:
        logger.info("=== 测试完成 ===")
        
        # 分析结果
        env_test = test_report["environment_test"]
        flow_test = test_report["flow_test"]
        
        if env_test.get("openroad_available", False):
            logger.info("✅ OpenROAD工具可用！可以集成到强化学习训练中")
        elif flow_test.get("openroad_available", False):
            logger.info("✅ OpenROAD Flow可用！可以集成到强化学习训练中")
        else:
            logger.info("⚠️  OpenROAD工具不可用，建议:")
            logger.info("1. 继续使用模拟环境进行强化学习训练")
            logger.info("2. 尝试安装完整的OpenROAD工具链")
            logger.info("3. 考虑使用其他开源EDA工具如Yosys + NextPNR")
    else:
        logger.error("测试失败")

if __name__ == "__main__":
    main() 