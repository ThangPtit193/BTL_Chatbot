import inspect
from typing import Any, Dict, Tuple, Callable

import mmh3

from functools import wraps
import time

import os

# System call
os.system("")


# Class of different styles
class Style:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {Style.CYAN}{func.__name__} {Style.RESET} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def args_to_kwargs(args: Tuple, func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    arg_names = list(sig.parameters.keys())
    # skip self and cls args for instance and class methods
    if any(arg_names) and arg_names[0] in ["self", "cls"]:
        arg_names = arg_names[1: 1 + len(args)]
    args_as_kwargs = {arg_name: arg for arg, arg_name in zip(args, arg_names)}
    return args_as_kwargs


def get_id(content):
    return "{:02x}".format(mmh3.hash128(str(content), signed=False))
