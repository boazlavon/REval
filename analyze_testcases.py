import json
import sys
from collections import defaultdict

def analyze_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    task_dict = defaultdict(lambda: {'input_idxs': set(), 'samples': set()})
    samples_dict  = defaultdict(lambda: {'samples': set()})
    
    for entry in data:
        task_idx = entry[0]
        input_idx = entry[1]
        task_dict[task_idx]['input_idxs'].add(input_idx)
        
        if len(entry) == 4:
            task_dict[task_idx]['samples'].add((input_idx, entry[2], entry[3]))
            samples_dict[(task_idx, input_idx)]['samples'].add((entry[2], entry[3]))
        else:
            task_dict[task_idx]['samples'].add((input_idx, entry[2]))
            samples_dict[(task_idx, input_idx)]['samples'].add((entry[3],))
    
    num_tasks = len(task_dict)
    tasks_set = list(task_dict.keys())
    avg_input_idxs_per_task = sum(len(v['input_idxs']) for v in task_dict.values()) / num_tasks
    total_samples = sum(len(v['samples']) for v in task_dict.values())
    avg_sample_per_task = sum(len(v['samples']) for v in task_dict.values()) / num_tasks
    avg_sample_per_task_idx = sum(len(v['samples']) for v in samples_dict.values()) / len(samples_dict.keys())
    
    result = {
        'num_tasks': num_tasks,
        'avg_input_idxs_per_task': avg_input_idxs_per_task,
        'avg_sample_per_task': avg_sample_per_task,
        'avg_sample_per_task_idx': avg_sample_per_task_idx,
        'total_samples': total_samples
    }
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_json.py <json_path>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    analysis = analyze_json(json_path)
    print(json.dumps(analysis, indent=4))
