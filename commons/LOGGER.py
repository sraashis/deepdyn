
class Logger:

    def __init__(self):
        print('Logging')

    @staticmethod
    def log(msg):
        print('[Log] ' + msg)

    @staticmethod
    def warn(msg):
        print('[Warn] ' + msg)

    @staticmethod
    def err(msg):
        print('[Err] msg ' + msg)