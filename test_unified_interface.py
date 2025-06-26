#!/usr/bin/env python3
"""
测试统一的Docker OpenROAD接口
"""

import subprocess
import logging
import os
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_openroad_with_docker(work_dir: Path, cmd: str, is_tcl: bool = True, timeout: int = 600) -> subprocess.CompletedProcess:
    """
    统一通过Docker调用OpenROAD
    :param work_dir: 挂载和工作目录
    :param cmd: TCL脚本文件名（只需文件名，不带路径），或直接openroad命令
    :param is_tcl: 是否为TCL脚本（True则自动拼接/workspace/xxx.tcl）
    :param timeout: 超时时间（秒）
    :return: subprocess.CompletedProcess对象
    """
    if is_tcl:
        cmd_in_container = f"/workspace/{cmd}"
        openroad_cmd = f"openroad {cmd_in_container}"
    else:
        openroad_cmd = f"openroad {cmd}"
    
    docker_cmd_str = f'docker run --rm -v {work_dir.absolute()}:/workspace -w /workspace openroad/flow-ubuntu22.04-builder:21e414 bash -c "export PATH=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\\$PATH && {openroad_cmd}"'
    
    logger.info(f"调用Docker OpenROAD: {openroad_cmd} @ {work_dir}")
    logger.info(f"完整命令: {docker_cmd_str}")
    
    return subprocess.run(docker_cmd_str, shell=True, capture_output=True, text=True, timeout=timeout)

def write_tcl_script(script_file: Path, content: str):
    """写入TCL脚本并确保文件同步到磁盘"""
    with open(script_file, 'w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    logger.info(f"✅ TCL脚本已写入并同步: {script_file}")

def test_simple_placement():
    """测试简单布局"""
    logger.info("=== 测试简单布局 ===")
    
    # 创建测试目录
    test_dir = Path("test_simple_placement")
    test_dir.mkdir(exist_ok=True)
    
    # 创建简单的TCL脚本
    simple_script = """
puts "开始简单布局测试..."
puts "当前工作目录: [pwd]"
puts "文件列表: [glob *]"
puts "简单布局测试完成"
"""
    
    script_file = test_dir / "simple_test.tcl"
    write_tcl_script(script_file, simple_script)
    
    try:
        logger.info("开始执行简单测试...")
        result = run_openroad_with_docker(test_dir, "simple_test.tcl", timeout=60)
        
        if result.returncode == 0 or "简单布局测试完成" in result.stdout:
            logger.info("✅ 简单测试执行成功")
            logger.info(f"输出: {result.stdout}")
            return True
        else:
            logger.error("❌ 简单测试执行失败")
            logger.error(f"错误: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 简单测试执行异常: {e}")
        return False

def test_version():
    """测试版本命令"""
    logger.info("=== 测试版本命令 ===")
    
    try:
        result = run_openroad_with_docker(Path.cwd(), "-version", is_tcl=False, timeout=30)
        if result.returncode == 0:
            logger.info("✅ 版本检查成功")
            logger.info(f"版本信息: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ 版本检查失败: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ 版本检查异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始统一接口测试...")
    
    tests = [
        ("版本命令", test_version),
        ("简单布局", test_simple_placement),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    logger.info("\n" + "="*60)
    logger.info("统一接口测试结果")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！统一接口工作正常")
    else:
        logger.warning("⚠️  部分测试失败，需要进一步调试")
    
    return passed == total

if __name__ == "__main__":
    main() 