from collections import namedtuple
from configparser import ConfigParser
import logging
from pathlib import Path
import psycopg2
import psycopg2.extras
import os
from typing import List, Iterable, Generic, Callable, TypeVar

from vfe.core.segment import Segment

T = TypeVar('T')

MakeTimestampArgs = namedtuple('MakeTimestampArgs', ['year', 'month', 'day', 'hour', 'min', 'sec'], defaults=[1] * 6)

def config(filename, section='postgresql'):
    filename=os.path.join(Path(__file__).absolute().parent, filename)
    logging.debug(f'Reading database configuration from {filename}')
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')
    return db


class DB:
    @staticmethod
    def _connect(db_config_file):
        conn = None
        try:
            params = config(filename=db_config_file)
            print('Connecting to the PostgreSQL database…')
            return psycopg2.connect(**params)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def __init__(self, config_file=Path(__file__).parent / 'database.ini'):
        self.conn = self._connect(config_file)
        # Create base tables.
        self._execute_stmts(self._create_table_statements())

    def add_media_object(self, object_name, dataset, split):
        self._commit("""
            INSERT INTO media_object (object_name, dataset, split)
            VALUES (%s, %s, %s)
            ON CONFLICT (object_name)
            DO UPDATE SET dataset=EXCLUDED.dataset, split=EXCLUDED.split
        """, (object_name, dataset, split))

    def add_media_segment(self, segment_name, object_name, path, make_timestamp_args: MakeTimestampArgs):
        self._commit("""
            INSERT INTO media_segment (object_id, segment_name, path, capture_time)
            VALUES (%s, %s, %s, make_timestamp(%s, %s, %s, %s, %s, %s))
        """, (self._get_object_id(object_name), segment_name, path, *make_timestamp_args))

    def add_metadata(self, segment_ids: List[int], label: str):
        cur = self.conn.cursor()
        psycopg2.extras.execute_values(cur, """
            INSERT INTO segment_metadata (segment_id, label)
            VALUES %s
            ON CONFLICT DO NOTHING
        """, [(s_id, label) for s_id in segment_ids])
        cur.close()
        self.conn.commit()

    def get_segments(self, object_ids) -> Iterable[Segment]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT segment_name, segment_id, path, capture_time, keyframe_path
            FROM media_segment
            WHERE object_id=ANY(%s)
        """, (object_ids,))
        def process_row(row) -> Segment:
            return Segment(*row)
        return DBIterator(cur, process_row)

    def _get_object_id(self, object_name):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT object_id
            FROM media_object
            WHERE object_name=%s
            LIMIT 1
        """, (object_name,))
        result = cur.fetchone()
        return None if not result else result[0]

    def _create_table_statements(self):
        return [
            """
            CREATE TABLE IF NOT EXISTS media_object (
                object_id SERIAL PRIMARY KEY,
                object_name VARCHAR(255) NOT NULL UNIQUE,
                dataset TEXT NOT NULL,
                split TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS media_segment (
                segment_id SERIAL PRIMARY KEY,
                object_id INTEGER NOT NULL,
                segment_name VARCHAR(255) NOT NULL UNIQUE,
                path VARCHAR(255) NOT NULL,
                keyframe_path VARCHAR(255) DEFAULT NULL,
                capture_time TIMESTAMP DEFAULT NULL,
                FOREIGN KEY (object_id) REFERENCES media_object (object_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS media_segment_object_idx
            ON media_segment (object_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS segment_metadata (
                segment_id INTEGER,
                label VARCHAR(255) NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES media_segment (segment_id),
                UNIQUE (segment_id, label)
            )
            """,
        ]

    def _execute_stmts(self, stmts):
        cur = self.conn.cursor()
        for stmt in stmts:
            cur.execute(stmt)
        cur.close()
        self.conn.commit()

    def _commit(self, stmt, vars=None):
        cur = self.conn.cursor()
        cur.execute(stmt, vars)
        cur.close()
        self.conn.commit()


class DBIterator(Generic[T]):
    def __init__(self, cursor, process_fn: Callable[..., T]):
        self.cursor = cursor
        self.process_fn = process_fn

    def __iter__(self):
        return self

    def __next__(self) -> T:
        row = self.cursor.fetchone()
        if row is None:
            raise StopIteration
        return self.process_fn(row)