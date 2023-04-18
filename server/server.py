import json
import logging
from pathlib import Path
import rpyc
from rpyc.utils.helpers import classpartial
from rpyc.utils.server import ThreadedServer

from vfe.api.storagemanager import LabelInfo
from vfe.utils.create_managers import get_alm

logger = logging.getLogger("__name__")

@rpyc.service
class VOCALExploreService(rpyc.Service):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alm = get_alm(config_path)

    def on_connect(self, conn):
        logger.info("Connected")

    @rpyc.exposed
    def add_video(self, video_path, start_time):
        logger.info('Adding video')
        self.alm.add_video(video_path, start_time)

    @rpyc.exposed
    def get_videos(self, limit=None):
        logger.info('Getting videos')
        vpaths_and_thumbnails = self.alm.get_videos(limit=limit, thumbnails=True)
        return vpaths_and_thumbnails

    @rpyc.exposed
    def get_labels(self, vid):
        logger.info("Getting labels")
        labels = self.alm.get_labels([vid])
        return {
            'labels': [label._asdict() for label in labels],
        }

    @rpyc.exposed
    def add_label(self, label_dict):
        logger.info("Adding label")
        label_info = LabelInfo(
            lid=None,
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label']
        )
        self.alm.add_labels([label_info])

    @rpyc.exposed
    def update_label(self, label_dict):
        logger.info("Updating label")
        label_info = LabelInfo(
            lid=label_dict['lid'],
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label'],
        )
        self.alm.update_labels([label_info])

    @rpyc.exposed
    def delete_label(self, label_dict):
        logger.info(f"Deleting label {label_dict['lid']}")
        label_info = LabelInfo(
            lid=label_dict['lid'],
            vid=label_dict['vid'],
            start_time=label_dict['start_time'],
            end_time=label_dict['end_time'],
            label=label_dict['label'],
        )
        self.alm.delete_labels([label_info])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_path = '/home/maureen/VOCALExplore/server/configs/r3d.yaml'
    # service = classpartial(VOCALExploreService, config_path=config_path)
    service = VOCALExploreService(config_path=config_path)
    t = ThreadedServer(
        service,
        port=8890,
        protocol_config={
            "allow_public_attrs": True,
        }
    )
    t.start()
