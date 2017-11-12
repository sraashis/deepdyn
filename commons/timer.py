import time

from commons.LOGGER import Logger


def check_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        Logger.log("------[ RUNNING Time: " + str(time.time()-start) + " seconds]------")
    return inner

