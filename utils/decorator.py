import tracemalloc
from functools import wraps
import time
from loguru import logger


def measure_memory(func):
    @wraps(func)
    def measure_memory_wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"\033[37mCurrent memory usage for {func.__name__}:\033[36m {current / 10 ** 6}\033[0mMB")
        return result

    return measure_memory_wrapper


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'\033[37mTime-consuming for {func.__name__}:\033[36m{total_time:.4f}\033[0m seconds')
        return result

    return timeit_wrapper


def instance(func):
    @wraps(func)
    def instance_wrapper(*args, **kwargs):
        pass
