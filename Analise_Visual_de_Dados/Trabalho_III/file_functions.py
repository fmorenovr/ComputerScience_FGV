#!/usr/bin/python

from pathlib import Path
import json
import glob
import shutil
import numpy as np


def openFile(filename):
  with open(filename) as f:
    config = json.load(f)
  
  return config

def get_db_names(path):
  files_path = glob.glob(path)
  files_path = np.sort(files_path).tolist()
  #files_name = [file.replace("../data/sql/crimebb-","").replace(f"{date}.sql","") for file in files_path]
  files_name = [(file_.split("/"))[-1].replace("crimebb-","").replace(f".sql","") for file_ in files_path]
  
  zip_iterator = zip(files_name, files_path)
  files_dict = dict(zip_iterator)
  
  files_path = [f"{Path.cwd().as_posix()}/{file_}" for file_ in files_path]
    
  return files_name, files_path, files_dict

def get_current_path():
    return str(Path(__file__).parent.resolve())

def get_absolute_path(relative_path):
    return Path(relative_path).absolute().as_posix()

def verifyFile(files_list):
    return Path(files_list).is_file()

def verifyType(file_name):
    if Path(file_name).is_dir():
        return "dir"
    elif Path(file_name).is_file():
        return "file"
    else:
        return None

def verifyDir(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, mode=0o777, exist_ok=True)