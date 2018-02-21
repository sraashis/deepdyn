import time


def checktime(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('-Function  "' + func.__name__ + '(__)"  took ' + str(round(time.time() - start, 3)) + " seconds-")
        return result

    return inner
