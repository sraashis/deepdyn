import os


def join(root, add):
    return os.path.join(root, add)


# Working directory path.(Project root folder)
CONTEXT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = join(CONTEXT_PATH, 'data')
OUT_PATH = join(CONTEXT_PATH, 'out')
CONF_PATH = join(CONTEXT_PATH, 'conf')
