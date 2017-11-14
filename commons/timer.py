import time

import logging as logger


def check_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.log("------[ RUNNING Time: " + str(time.time() - start) + " seconds ]------")
        return result
    return inner

