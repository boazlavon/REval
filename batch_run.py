import os
import sys
import subprocess
import shlex

from tqdm import tqdm

model_id = sys.argv[1]
num_repeats = 5
config_path = '.configs'
save_output_path = '.batch_run'
python_path = '/home/zhiyuanpan/anaconda3/envs/dreval/bin/python'
tasks = ['coverage', 'state', 'path', 'output']

os.makedirs(save_output_path, exist_ok=True)

def get_cmd(task, run_id):
    return f'{python_path} evaluation.py -i {config_path}/{model_id}_{task} > {save_output_path}/{model_id}_{task}_run{run_id}'

for i in tqdm(range(num_repeats)):
    procs = []
    for task in tasks:
        proc = subprocess.Popen(get_cmd(task, i+1), shell=True)
        procs.append(proc)
        print(f'Spawned task {task} for model {model_id} with pid {proc.pid} ({i+1}/{num_repeats})')
    for idx, proc in enumerate(procs):
        proc.wait()
        print(f'Finished task {tasks[idx]} for model {model_id} ({i+1}/{num_repeats})')
    # deal with `consistency` at last
    proc = subprocess.Popen(get_cmd('consistency', i+1), shell=True)
    proc.wait()
    print(f'Finished task consistency for model {model_id} ({i+1}/{num_repeats})')
