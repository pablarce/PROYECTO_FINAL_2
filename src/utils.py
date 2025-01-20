import os

def create_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
