import logging

def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(processName)s] [%(threadName)s] %(asctime)s %(name)s %(funcName)s: %(message)s',
        force=True,
    )
