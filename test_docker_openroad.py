#!/usr/bin/env python3
"""
测试Docker和OpenROAD的可用性
"""

import subprocess
import sys
import os
from pathlib import Path

def test_docker():
    """测试Docker是否可用"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Docker可用: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Docker不可用: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Docker测试失败: {e}")
        return False

def test_openroad_image():
    """测试OpenROAD镜像是否可用"""
    try:
        # 检查镜像是否存在
        result = subprocess.run(['docker', 'images', 'openroad/flow-ubuntu22.04-builder:21e414'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'openroad/flow-ubuntu22.04-builder' in result.stdout:
            print("✅ OpenROAD镜像存在")
            return True
        else:
            print("❌ OpenROAD镜像不存在")
            return False
    except Exception as e:
        print(f"❌ OpenROAD镜像测试失败: {e}")
        return False

def test_openroad_container():
    """测试OpenROAD容器是否能正常运行"""
    try:
        # 创建一个简单的测试命令，设置正确的PATH
        test_cmd = [
            'docker', 'run', '--rm',
            'openroad/flow-ubuntu22.04-builder:21e414',
            'bash', '-c', 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad --version'
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and 'OpenROAD' in result.stdout:
            print(f"✅ OpenROAD容器测试成功: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ OpenROAD容器测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ OpenROAD容器测试异常: {e}")
        return False

def test_simple_tcl():
    """测试简单的TCL脚本执行"""
    try:
        # 创建一个简单的TCL脚本
        simple_tcl = """
puts "Hello from OpenROAD TCL"
puts "Current directory: [pwd]"
puts "Testing basic TCL functionality"
"""
        
        # 将TCL脚本写入临时文件
        tcl_file = Path("test_simple.tcl")
        with open(tcl_file, 'w') as f:
            f.write(simple_tcl)
        
        # 在Docker容器中执行TCL脚本，设置正确的PATH
        test_cmd = [
            'docker', 'run', '--rm', '-v', f'{os.getcwd()}:/workspace',
            'openroad/flow-ubuntu22.04-builder:21e414',
            'bash', '-c', 'export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH && openroad -exit /workspace/test_simple.tcl'
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ TCL脚本测试成功:")
            print(result.stdout)
            return True
        else:
            print(f"❌ TCL脚本测试失败:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ TCL脚本测试异常: {e}")
        return False
    finally:
        # 清理临时文件
        if tcl_file.exists():
            tcl_file.unlink()

def main():
    """主函数"""
    print("=== Docker和OpenROAD可用性测试 ===\n")
    
    # 测试Docker
    docker_ok = test_docker()
    print()
    
    # 测试OpenROAD镜像
    image_ok = test_openroad_image()
    print()
    
    # 测试OpenROAD容器
    container_ok = test_openroad_container()
    print()
    
    # 测试TCL脚本
    tcl_ok = test_simple_tcl()
    print()
    
    # 总结
    print("=== 测试总结 ===")
    print(f"Docker: {'✅' if docker_ok else '❌'}")
    print(f"OpenROAD镜像: {'✅' if image_ok else '❌'}")
    print(f"OpenROAD容器: {'✅' if container_ok else '❌'}")
    print(f"TCL脚本: {'✅' if tcl_ok else '❌'}")
    
    if all([docker_ok, image_ok, container_ok, tcl_ok]):
        print("\n🎉 所有测试通过！Docker和OpenROAD环境正常。")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 