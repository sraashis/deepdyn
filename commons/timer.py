import time


def check_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("------[ RUNNING Time: " + str(time.time() - start) + " seconds ]------")
        return result

    return inner
