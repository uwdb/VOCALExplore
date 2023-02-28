import os

def create_dir(*args):
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def ensure_exists(path):
    if not os.path.exists(path):
        create_dir(os.path.dirname(path))
        with open(path, 'w+') as f:
            pass
