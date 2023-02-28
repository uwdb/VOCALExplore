import argparse
import json
import glob
import os
import re

def parse_file(path):
    info = {}
    # step = -1
    in_bandit = False
    with open(path, 'r') as f:
        for line in f:
            if 'before pruning, candidates are' in line:
                # New step; overwrite info.
                info = {}
                in_bandit = True
                # step = int(re.search(r'After step (\d+) ', line)[1])
            elif 'after pruning, candidates are' in line:
                in_bandit = False
            elif in_bandit and '_prune' in line and 'Removing candidate' not in line:
                try:
                    feature_info = line.split('_prune:')[1].strip()
                    feature, data = feature_info.split(': ')
                    for key in ['y_k', 'w_k', 'l_k', 'u_k']:
                        data = data.replace(f'{key}=', f'"{key}": ')
                    info[feature] = json.loads('{' + data + '}')
                except:
                    return None
    return info

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir')
    ap.add_argument('--target-suffix', default='bandit*.txt')
    args = ap.parse_args()

    log_dir = args.log_dir
    json_dir = log_dir.replace('/logs', '/json')
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    for path in glob.glob(os.path.join(log_dir, f'*{args.target_suffix}')):
        out_path = path.replace(log_dir, json_dir).replace('.txt', '.json')
        if os.path.exists(out_path):
            continue
        info = parse_file(path)
        if info:
            with open(out_path, 'w+') as outf:
                json.dump(info, outf)
