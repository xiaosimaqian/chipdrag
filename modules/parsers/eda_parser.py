import re
import json
from pathlib import Path
from typing import Dict, Any

def parse_def(def_file_path: str) -> Dict[str, Any]:
    """
    A simple DEF parser to extract key metrics like DIEAREA and component count.
    This uses regular expressions and is not a full-fledged DEF parser.
    """
    metrics = {
        'die_area': None,
        'num_components': 0,
        'components_summary': {},
    }
    def_path = Path(def_file_path)
    if not def_path.exists():
        print(f"Error: DEF file not found at {def_file_path}")
        return metrics

    with open(def_path, 'r') as f:
        content = f.read()

        # Parse Die Area
        diearea_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
        if diearea_match:
            x1, y1, x2, y2 = map(int, diearea_match.groups())
            # Assuming DBU is 1000 for micron conversion. This might need adjustment.
            dbu_per_micron = 1000
            metrics['die_area'] = (x2 - x1, y2 - y1)
            metrics['die_area_microns'] = ((x2 - x1) / dbu_per_micron, (y2 - y1) / dbu_per_micron)

        # Parse Components
        components_match = re.search(r'COMPONENTS\s+(\d+)\s*;(.*?)END COMPONENTS', content, re.DOTALL)
        if components_match:
            num_components = int(components_match.group(1))
            metrics['num_components'] = num_components
            
            components_str = components_match.group(2)
            # Find all component types
            # Line format: - compName compType + ...
            lines = components_str.strip().split('\n')
            for line in lines:
                if line.strip().startswith('-'):
                    parts = line.split()
                    if len(parts) >= 3:
                        comp_type = parts[2]
                        metrics['components_summary'][comp_type] = metrics['components_summary'].get(comp_type, 0) + 1
    return metrics

def parse_verilog(verilog_file_path: str) -> Dict[str, Any]:
    """
    A simple Verilog parser to extract key metrics using regex.
    This is a simplified parser and may not be accurate for all Verilog styles.
    """
    metrics = {
        'num_modules': 0,
        'num_inputs': 0,
        'num_outputs': 0,
        'num_wires': 0,
        'num_instances': 0,
    }
    v_path = Path(verilog_file_path)
    if not v_path.exists():
        print(f"Error: Verilog file not found at {verilog_file_path}")
        return metrics

    with open(v_path, 'r', errors='ignore') as f:
        content = f.read()
        # This is a very basic count and might not be fully accurate.
        metrics['num_modules'] = len(re.findall(r'^\s*module\s+', content, re.MULTILINE))
        metrics['num_inputs'] = len(re.findall(r'^\s*input\s+', content, re.MULTILINE))
        metrics['num_outputs'] = len(re.findall(r'^\s*output\s+', content, re.MULTILINE))
        metrics['num_wires'] = len(re.findall(r'^\s*wire\s+', content, re.MULTILINE))
        # Count cell instantiations (simplified, assumes syntax like `CELL_TYPE instance_name (...)`)
        # This regex avoids matching module declarations
        metrics['num_instances'] = len(re.findall(r'^\s*([a-zA-Z_][\w_]*)\s+([a-zA-Z_][\w_]*)\s*\(', content, re.MULTILINE))
        
    return metrics

if __name__ == '__main__':
    # This block allows us to test the parser directly
    print("--- EDA Parser Test ---")
    
    netlist_path = "/Users/keqin/Documents/workspace/chip-rag/chipdrag/data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/design.v"
    def_path_user = "/Users/keqin/Documents/workspace/chip-rag/chipdrag/data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/mgc_des_perf_1_place.def"
    
    print(f"\n--- Parsing Verilog File: {netlist_path} ---")
    v_metrics = parse_verilog(netlist_path)
    if v_metrics.get('num_instances', 0) > 0:
        print(json.dumps(v_metrics, indent=2))
    else:
        print("Could not parse Verilog metrics.")

    print(f"\n--- Parsing DEF File provided by user: {def_path_user} ---")
    def_metrics = parse_def(def_path_user)
    
    # Suggesting the correct file if user-provided one fails
    if not def_metrics.get('die_area'):
        print("\nCould not parse the specified DEF file. It might not exist or be in the wrong format.")
        def_path_alt = "/Users/keqin/Documents/workspace/chip-rag/chipdrag/data/designs/ispd_2015_contest_benchmark/mgc_des_perf_1/floorplan.def"
        print(f"Let's try parsing an alternative file found in the directory: {def_path_alt}")
        
        print(f"\n--- Parsing Alternative DEF File: {def_path_alt} ---")
        def_metrics_alt = parse_def(def_path_alt)
        if def_metrics_alt.get('die_area'):
            print(json.dumps(def_metrics_alt, indent=2))
        else:
            print("Failed to parse the alternative DEF file as well.")
    else:
        print(json.dumps(def_metrics, indent=2)) 