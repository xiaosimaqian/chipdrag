#!/usr/bin/env python3
"""
自动删除DEF文件中的ROWS区块
"""
from pathlib import Path
import re

def remove_rows_block(def_path):
    with open(def_path, 'r') as f:
        content = f.read()
    # 删除ROWS ... END ROWS区块（支持多行）
    new_content, n = re.subn(r'ROWS.*?END ROWS\s*', '', content, flags=re.DOTALL)
    if n > 0:
        backup = def_path + '.bak_rows'
        with open(backup, 'w') as f:
            f.write(content)
        with open(def_path, 'w') as f:
            f.write(new_content)
        print(f"✅ 已删除ROWS区块，原文件备份为: {backup}")
    else:
        print("未找到ROWS区块，无需修改。")

def main():
    def_file = "data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def"
    remove_rows_block(def_file)

if __name__ == "__main__":
    main() 