# TIDES/scripts/analyze_results.py
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir):
    results = []
    results_dir = Path(results_dir)
    for result_file in sorted(results_dir.glob('*.json')):
        with open(result_file, 'r') as f:
            results.append(json.load(f))
    return results

def analyze_results(results_dir):
    """Analyze results"""
    results = load_results(results_dir)
    
    stats = {
        'total_queries': len(results),
        'avg_time': sum(r['time'] for r in results) / len(results),
        'retrieval_success': sum(1 for r in results if r['stage1']),
        'answer_success': sum(1 for r in results if r['stage3'] != 'No answer')
    }
    
    df = pd.DataFrame(results)
    
    return stats, df

    

def main():
    parser = argparse.ArgumentParser(description='Analyze TIDES results')
    parser.add_argument('--results-dir', required=True,
                      help='Directory containing result files')
    parser.add_argument('--output-dir', default='analysis',
                      help='Directory for analysis outputs')
    args = parser.parse_args()
    
    stats, df = analyze_results(args.results_dir)
    print("Analysis Results:")
    print(json.dumps(stats, indent=2))
    

if __name__ == '__main__':
    main()