import duckdb
import pandas as pd

def load_classes():
    activities = set()
    with open('in_700_not_400-filtered.txt', 'r') as f:
        for line in f:
            activities.add(line.strip())
    return activities

def create_split(split, activities):
    og_csv = pd.read_csv(f'/data/kinetics700/{split}.csv')
    subset_csv = og_csv[og_csv.label.isin(activities)]
    # Filter to rows where the vid isn't in k400.
    k400_csv = pd.read_csv(f'/data/kinetics400/annotations/{split}.csv')
    k400_ids = k400_csv.youtube_id.unique()
    subset_csv_filtered = subset_csv[~subset_csv.youtube_id.isin(k400_ids)]
    # Filter out duplicate rows.
    con = duckdb.connect()
    con.execute("""
        COPY (
            SELECT label, youtube_id, time_start, time_end, split
            FROM subset_csv_filtered
            GROUP BY ALL
        ) TO '/gscratch/balazinska/mdaum/data/kinetics700-400/{split}.csv'
        (HEADER, DELIMITER ',')
    """.format(split=split))
    # subset_csv_filtered.to_csv(f'/gscratch/balazinska/mdaum/data/kinetics700-400/{split}.csv', index=False)

def create_videometadatacsv(split):
    info_csv = pd.read_csv(f'/gscratch/balazinska/mdaum/data/kinetics700-400/{split}.csv')
    info_csv['path'] = f'/data/kinetics700/{split}/' + info_csv.label + '/' + info_csv.youtube_id + '_'  + info_csv.time_start.astype(str).str.pad(width=6, side='left', fillchar='0') + '_' + info_csv.time_end.astype(str).str.pad(width=6, side='left', fillchar='0') + '.mp4'
    info_csv['start'] = None
    info_csv['duration'] = 10
    info_csv[['path', 'start', 'duration']].to_csv(f'/gscratch/balazinska/mdaum/data/kinetics700-400/{split}_videometadata.csv', index=False)

def create_annotationscsv(split):
    info_csv = pd.read_csv(f'/gscratch/balazinska/mdaum/data/kinetics700-400/{split}.csv')
    info_csv['path'] = f'/data/kinetics700/{split}/' + info_csv.label + '/' + info_csv.youtube_id + '_'  + info_csv.time_start.astype(str).str.pad(width=6, side='left', fillchar='0') + '_' + info_csv.time_end.astype(str).str.pad(width=6, side='left', fillchar='0') + '.mp4'
    info_csv['start'] = 0
    info_csv['end'] = 10
    info_csv[['path', 'start', 'end', 'label']].to_csv(f'/gscratch/balazinska/mdaum/data/kinetics700-400/{split}_annotations.csv', index=False)

if __name__ == '__main__':
    activities = load_classes()
    for split in ['train', 'val']:
        create_split(split, activities)
        create_videometadatacsv(split)
        create_annotationscsv(split)
