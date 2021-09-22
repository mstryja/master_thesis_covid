from genericpath import isdir
import os
import json
import pickle
from os.path import join
from json import dump, load
# from posix import listdir


def list_dirs(path: str):
    return [dir for dir in os.listdir(path) if os.path.isdir(join(path, dir))]


def train_dir(path: str):
    subdirs = list_dirs(path)
    for dir in subdirs:
        if 'train' in dir:
            return os.path.join(path, dir)
    return None


def test_dir(path: str):
    subdirs = list_dirs(path)
    for dir in subdirs:
        if 'test' in dir:
            return os.path.join(path, dir)
    return None


def list_train_images(path: str):
    train_path = train_dir(path)
    train = list_dirs(train_path)
    exams = []
    for case in train:
        subcase = list_dirs(join(train_path, case))[0]
        exams.append(join(train_path, case, subcase, list_spec_files(join(train_path, case, subcase), '.dcm')[0]))

    return exams


def list_spec_files(path: str, spec: str):
    return [f for f in os.listdir(path) if f.endswith(spec)]


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

