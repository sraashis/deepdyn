
class Logger:

    def __init__(self):
        print('Logging')

    @staticmethod
    def log(self, msg):
        print('[Log] msg')

    @staticmethod
    def warn(self, msg):
        print('[Warn] msg')

    @staticmethod
    def err(self, msg):
        print('[Err] msg')