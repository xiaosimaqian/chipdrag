import json
from pathlib import Path

def generate_test_data(output_dir: str):
    """生成测试数据集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试用例1: 简单的ALU设计
    alu_netlist = {
        'modules': [{
            'name': 'alu',
            'type': 'system',
            'children': [
                {
                    'name': 'adder',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'adder_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'adder_cell1', 'type': 'cell'},
                                {'name': 'adder_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                },
                {
                    'name': 'multiplier',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'mult_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'mult_cell1', 'type': 'cell'},
                                {'name': 'mult_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                }
            ]
        }]
    }
    
    # 测试用例2: FFT设计
    fft_netlist = {
        'modules': [{
            'name': 'fft',
            'type': 'system',
            'children': [
                {
                    'name': 'butterfly',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'bf_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'bf_cell1', 'type': 'cell'},
                                {'name': 'bf_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                },
                {
                    'name': 'twiddle',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'tw_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'tw_cell1', 'type': 'cell'},
                                {'name': 'tw_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                }
            ]
        }]
    }
    
    # 测试用例3: 矩阵乘法设计
    matrix_netlist = {
        'modules': [{
            'name': 'matrix_mult',
            'type': 'system',
            'children': [
                {
                    'name': 'mac',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'mac_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'mac_cell1', 'type': 'cell'},
                                {'name': 'mac_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                },
                {
                    'name': 'accumulator',
                    'type': 'function',
                    'children': [
                        {
                            'name': 'acc_core',
                            'type': 'submodule',
                            'children': [
                                {'name': 'acc_cell1', 'type': 'cell'},
                                {'name': 'acc_cell2', 'type': 'cell'}
                            ]
                        }
                    ]
                }
            ]
        }]
    }
    
    # 保存测试数据
    test_cases = {
        'mgc_des_perf_1': alu_netlist,
        'mgc_fft_1': fft_netlist,
        'mgc_matrix_mult_1': matrix_netlist
    }
    
    for case_name, netlist in test_cases.items():
        case_dir = output_dir / case_name
        case_dir.mkdir(exist_ok=True)
        
        # 保存netlist
        with open(case_dir / 'netlist.json', 'w') as f:
            json.dump(netlist, f, indent=2)
        
        # 保存ground truth (与netlist相同)
        with open(case_dir / 'ground_truth.json', 'w') as f:
            json.dump(netlist, f, indent=2)

if __name__ == '__main__':
    generate_test_data('chiprag/dataset/benchmarks') 