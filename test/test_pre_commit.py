import logging
import time
import functools
import traceback

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s - %(lineno)d -msg: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def retry(retry_num, retry_sleep_sec):
    """
    retry help decorator.
    :param retry_num: the retry num; retry sleep sec
    :return: decorator
    """
    def decorator(func):
        """decorator"""
        # preserve information about the original function, or the func name will be "wrapper" not "func"
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """wrapper"""
            for attempt in range(retry_num):
                try:
                    return func(*args, **kwargs)  # should return the raw function's return value
                except Exception as err:   # pylint: disable=broad-except
                    logging.error(err)
                    logging.error(traceback.format_exc())
                    time.sleep(retry_sleep_sec)
                logging.error("Trying attempt %s of %s.", attempt + 1, retry_num)
            logging.error("func %s retry failed", func)
            raise Exception('Exceed max retry num: {} failed'.format(retry_num))

        return wrapper
    return decorator
