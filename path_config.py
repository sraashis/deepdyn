import os


def join(root, add):
    return os.path.join(root, add)


# Working directory path.(Project root folder)

path = os.path.dirname(os.path.abspath(__file__))
input_path = join(path, 'data')
output_path = join(path, 'out')
kernel_dmp_path = join(output_path, 'kern')
graph_dmp_path = join(output_path, 'graph')
conf_path = join(path, 'conf')
av_wide_data = join(input_path, 'av_wide_data_set')


def set_cwd(current_path):
    os.chdir(current_path)


def get_cwd():
    return os.getcwd()
