import logging
import os
from typing import Callable

from vfe.core import database

def ingest(path: str,
        is_segment_fn: Callable[[str], bool],
        object_fn: Callable[[str], str],
        dataset_fn: Callable[[str], str],
        split_fn: Callable[[str], str],
        segment_fn: Callable[[str], str],
        capturetime_fn: Callable[[str], database.MakeTimestampArgs],
        db_config_file=None) -> None:
    '''
    Walk directory to ingest information about media objects and segments.

    Uses object_fn and segment_fn to recognize media objects and corresponding segments and inserts them into the database.

    Parameters:
    path -- top-level of directory to walk
    is_segment_fn -- given a path return a Boolean whether it is a segment
    object_fn -- given a path return the media object's name
    dataset_fn -- given a path return the media object's dataset
    split_fn -- given a path return the media object's split
    segment_fn -- given a path return the segment object's name
    capturetime-fn -- given a path return the segment's capturetime
    '''
    logging.info('Ingesting media objects and segments into the database')

    db_args = {} if not db_config_file else {'config_file': db_config_file}
    db = database.DB(**db_args)

    seen_media_objects = set()
    num_segments = 0
    # TODO: Should we drop/re-add indexes on media_segment before adding all of the segments?
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            if not is_segment_fn(full_path):
                continue
            obj_name = object_fn(full_path)
            if not obj_name in seen_media_objects:
                logging.info(f'Found object {obj_name}')
                db.add_media_object(obj_name, dataset_fn(full_path), split_fn(full_path))
                seen_media_objects.add(obj_name)
            segment_name = segment_fn(full_path)
            assert len(segment_name) < 255
            db.add_media_segment(segment_name, obj_name, full_path, capturetime_fn(full_path))
    logging.info(f'Ingested {len(seen_media_objects)} objects and {num_segments} segments')
    return seen_media_objects
