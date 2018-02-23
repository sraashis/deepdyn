import time


def checktime(func):
    def inner(*args, **kwargs):
        start = time.time()
        print('"' + func.__name__ + '(__)" ...', end='')
        result = func(*args, **kwargs)
        print(' took ' + str(
            round(time.time() - start, 3)) + " seconds")
        return result

    return inner
