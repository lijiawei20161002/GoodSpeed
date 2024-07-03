import os
import shutil

def remove_cache_dirs_and_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove __pycache__ directories
        if '__pycache__' in dirnames:
            pycache_path = os.path.join(dirpath, '__pycache__')
            print(f"Removing directory {pycache_path}")
            shutil.rmtree(pycache_path)
        
        # Remove files ending with .lp and .sol
        for filename in filenames:
            if filename.endswith('.lp') or filename.endswith('.sol') or filename.endswith('.solver'):
                file_path = os.path.join(dirpath, filename)
                print(f"Removing file {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    root_directory = '.'  
    remove_cache_dirs_and_files(root_directory)