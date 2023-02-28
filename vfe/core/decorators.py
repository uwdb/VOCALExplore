import logging
import time
from functools import partial

def log_performance(func=None, param=None):
    if func is None:
        return partial(log_performance, param=param)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        return_val = func(*args, **kwargs)
        end = time.perf_counter()
        logging.debug(f'perf: {func.__name__} took {end - start} sec')
        return return_val
    return wrapper
