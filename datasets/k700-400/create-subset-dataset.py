import argparse
import duckdb
import glob
import os
from pathlib import Path
import shutil

from vfe import core

def copy_db(classpath, base_dir, target_dir, clean=False):
    if clean and os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    if not os.path.exists(target_dir):
        shutil.copytree(base_dir, target_dir)

    con = duckdb.connect(str(target_dir / 'annotations.duckdb'))
    con.execute("""
        CREATE TEMP TABLE keeplabels AS
        SELECT * FROM read_csv_auto('{classpath}', HEADER=False, columns={{'label': 'VARCHAR'}})
    """.format(classpath=classpath))
    con.execute("""
        DROP TABLE models;
    """)
    con.execute("""
        CREATE TEMP TABLE keepvids AS
        SELECT vid FROM annotations WHERE label in (
            SELECT * FROM keeplabels
        )
    """)
    before_annotations = con.execute("select count(*) from annotations").fetchall()[0][0]
    before_videometadata = con.execute("select count(*) from video_metadata").fetchall()[0][0]
    print(f'Before: {before_annotations} annotations, {before_videometadata} metadata')
    con.execute("""
        DELETE FROM annotations WHERE vid NOT IN (
            SELECT vid FROM keepvids
        )
    """)
    con.execute("""
        DELETE FROM video_metadata WHERE vid NOT IN (
            SELECT vid FROM keepvids
        )
    """)
    after_annotations = con.execute("select count(*) from annotations").fetchall()[0][0]
    after_videometadata = con.execute("select count(*) from video_metadata").fetchall()[0][0]
    print(f'After: {after_annotations} annotations, {after_videometadata} metadata')

    features_dir = target_dir / 'features'
    og_features_dir = target_dir / 'features_og'
    os.rename(features_dir, og_features_dir)
    copy_features(con, features_dir, og_features_dir)
    shutil.rmtree(str(og_features_dir))

def copy_features(con, target_dir, og_features_dir, feature_names=[]):
    if not feature_names:
        feature_names = os.listdir(og_features_dir)
    for feature_name in feature_names:
        con.execute("""
            CREATE TEMP TABLE features AS
            SELECT * FROM read_parquet('{feature_path}/*.parquet')
            WHERE vid IN (
                SELECT * FROM keepvids
            )
        """.format(feature_path=str(og_features_dir / feature_name)))
        core.filesystem.create_dir(target_dir / feature_name)
        con.execute("""
            COPY (SELECT * FROM features) TO '{feature_path}/0-0.parquet' (FORMAT 'parquet')
        """.format(feature_path=str(target_dir / feature_name)))
        con.execute("""
            DROP TABLE features
        """)
        for fname in ['_fid.txt', '_version.txt']:
            shutil.copy(og_features_dir / feature_name / fname, target_dir / feature_name / fname)

def copy_features_only(base_train, subset_train, feature_names):
    con = duckdb.connect(str(subset_train / 'annotations.duckdb'))
    features_dir = subset_train / 'features'
    og_features_dir = base_train / 'features'
    con.execute("""
        CREATE TEMP TABLE keepvids AS
        SELECT vid FROM annotations
    """)
    copy_features(con, features_dir, og_features_dir, feature_names)
    con.execute("""
        DROP TABLE keepvids
    """)

def dump_db(db_path, dump_dir, clean=False):
    exists = os.path.exists(dump_dir)
    if exists and not clean:
        return
    if exists:
        shutil.rmtree(dump_dir)
    con = duckdb.connect(str(db_path / 'annotations.duckdb'), read_only=True)
    con.execute(f"EXPORT DATABASE '{dump_dir}'")
    con.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--classpath')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--features-only', action='store_true')
    ap.add_argument('--features', nargs='+')
    args = ap.parse_args()

    base = Path('/gscratch/balazinska/mdaum/video-features-exploration/service/storage')
    base_train = base / 'kinetics7m4-train/oracle'
    base_val = base / 'kinetics7m4-val'

    classpath = Path(args.classpath)
    name = os.path.splitext(classpath.name)[0]
    subset_train = base / f'{name}-train/oracle'
    subset_val = base / f'{name}-val'

    if args.features_only:
        copy_features_only(base_train, subset_train, args.features)
        copy_features_only(base_val, subset_val, args.features)
    else:
        copy_db(classpath, base_train, subset_train, args.clean)
        dump_db(subset_train, str(subset_train).replace('oracle', 'oracle-dump'), args.clean)
        copy_db(classpath, base_val, subset_val, args.clean)
