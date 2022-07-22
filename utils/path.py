import os

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)