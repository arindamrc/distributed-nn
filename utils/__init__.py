import os, sys

def signum(n):
    if n >= 0:
        return 1
    else:
        return -1

def get_current_folder():
    return os.path.dirname(os.path.abspath(sys.argv[0]))

def get_timestamp_folder(experiment_timestamp):
    current_folder = get_current_folder()
    timestamp_folder = os.path.join(current_folder, experiment_timestamp)
    
    return timestamp_folder

def create_timestamp_folder(experiment_timestamp):
    timestamp_folder = get_timestamp_folder(experiment_timestamp)
    os.mkdir(timestamp_folder)
    
    return timestamp_folder