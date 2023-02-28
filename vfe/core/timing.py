import datetime
import logging
import time

TIME_LOGGER = logging.getLogger(__name__)

def logtime(func):
    def wrap_func(*args, **kwargs):
        TIME_LOGGER.info(f'{func.__name__} called at {datetime.datetime.now()}')
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        TIME_LOGGER.info(f'{func.__name__} took {end - start} seconds')
        return result
    return wrap_func
