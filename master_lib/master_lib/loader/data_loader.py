from os import pathsep
import nibabel as nib
import numpy as np
import pydicom as pdcm
import pandas as pd
from master_lib.utils import files_and_directories
from typing import List, Tuple
import random


JSON_DATASET_PATH = R'C:\Users\MikołajStryja\Documents\Studia\master\dataset.json'
DATASET_JSON = files_and_directories.load_json(JSON_DATASET_PATH)

def read_nifit(nifti_path: str):
    nifti_img = nib.load(nifti_path)
    return nifti_img


def get_nifti_data(nifti_img):
    arr = nifti_img.get_fdata()
    affine = nifti_img.affine()
    return arr, affine


def load_dicom(path: str):
    dicom = pdcm.dcmread(path)
    return dicom


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate_dataset_to_prediction(path_to_dataset_file: str, img_dataset_path: str, columns = None): 
    """
    Docstring must be written
    """
    train_paths = files_and_directories.list_train_images(img_dataset_path)
    dataset = files_and_directories.load_json(path_to_dataset_file)

    if columns is None:
        # Deafult, the prediction is making based on the four class - without bboxes:
        cols = ['ID', 'Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'Typical Appearance']
        df = pd.DataFrame(columns=cols)
        for key, val in dataset.items():
            values_list = [val[x] for x in cols[1:]]
            values_list.insert(0, key)
            temp_df = pd.DataFrame([values_list], columns=cols)
            df = pd.concat([df, temp_df], ignore_index=True)

    return df


def evaluate_categories(ID: str, cols: List = ["Atypical Appearance", "Indeterminate Appearance", "Negative for Pneumonia", "Typical Appearance"]):
    """
    Docstring must be written
    """
    evaluated_case = DATASET_JSON[ID]
    categories = []
    for col in cols:
        categories.append(evaluated_case[col])

    return np.array(categories)


def dataset_partition(train_images_path_list: List, coef: float = 0.3, evaluation_cases: int = 150, verbose: int = 1):
    max_idx = len(train_images_path_list)-evaluation_cases
    up = int(np.ceil(max_idx*(1 - coef)))
    if verbose==1:
        o = f"dataset length: {len(train_images_path_list)}\nEvaluation Cases: {evaluation_cases}"
        o += f"\nTrain cases: {up}\nValidation cases: {max_idx - up}"
        print(o)
    return train_images_path_list[:up], train_images_path_list[up:max_idx], train_images_path_list[max_idx:]
        



