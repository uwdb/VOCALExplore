import argparse
import duckdb
from fractions import Fraction
import math
import numpy as np
import os

from vfe import core

def create_probs_zipf(N, s=1):
    numerators = [Fraction(1, (n+1)**int(s)) for n in range(N)]
    denominator = sum(numerators)
    return numerators
    # Normalized. Don't do this so that the first class keeps all of its vids.
    # return [numerator / denominator for numerator in numerators]

def create_probs_exp(N, l):
    return [math.exp(-1 * l * x) for x in range(N)]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-type', default='zipf')
    ap.add_argument('--s', type=float, default=1)
    ap.add_argument('--base-dir', default='/gscratch/balazinska/mdaum/video-features-exploration/service/storage/kinetics7m4-train')
    args = ap.parse_args()

    base_dir = args.base_dir

    con = duckdb.connect(os.path.join(base_dir, 'oracle', 'annotations.duckdb'), read_only=True)
    all_classes = con.execute("SELECT DISTINCT label FROM annotations ORDER BY label").fetchnumpy()['label']

    N = len(all_classes)
    s = args.s
    split_type = f'{args.split_type}-N{N}-s{s}'
    splits_dir = os.path.join(base_dir, 'oracle-splits')

    probs = create_probs_zipf(N, s) if args.split_type == 'zipf' else create_probs_exp(N, s)
    rng = np.random.default_rng(5)
    for split_idx in range(10):
        # Shuffle classes.
        class_order = rng.permutation(N)

        split_file = os.path.join(splits_dir, split_type, f'split-{split_idx}', 'train.csv')
        core.filesystem.ensure_exists(split_file)
        with open(split_file, 'w+') as f:
            print('vpath', file=f)
            # For each class, select probs[i] of the vids from that class to keep.
            # Probabilities are always iterated over in decreasing order, but the order
            # of the classes changes.
            for i, class_idx in enumerate(class_order):
                label = all_classes[class_idx]
                prob = probs[i]
                all_vids = con.execute("""
                    SELECT DISTINCT vpath
                    FROM video_metadata v, annotations a
                    WHERE v.vid=a.vid
                        AND label=?
                """, [label]).fetchnumpy()['vpath']
                size = math.ceil(float(prob) * len(all_vids))
                sampled_vpaths = rng.choice(all_vids, size=size, replace=False)
                print(f'Prob {prob}, label {label}, nvids {len(sampled_vpaths)}')
                print(*sampled_vpaths, sep='\n', file=f)
