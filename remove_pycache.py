import os
import shutil

def remove_pycache_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            print(f"Removing {pycache_path}")
            shutil.rmtree(pycache_path)

if __name__ == "__main__":
    root_directory = '.'  
    remove_pycache_dirs(root_directory)