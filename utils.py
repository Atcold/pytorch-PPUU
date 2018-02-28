import torch
import os, json, pdb
from datetime import datetime

# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
            os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

