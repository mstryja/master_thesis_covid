import json
from matplotlib.pyplot import box
import pandas as pd
import numpy as np
from pandas.core.series import Series
from typing import List
import ast
from master_lib.loader import data_loader
from master_lib.utils import files_and_dirctories


def get_dict_image_level(dataset_case: Series, columns: List):
    id = columns.index('id')
    boxes = columns.index('boxes')
    label = columns.index('label')
    study_instance = columns.index('StudyInstanceUID')

    id = dataset_case[id].split('_')[0] + '.dcm'
    if pd.isnull(dataset_case[boxes]):
        boxes = {}
        label = dict_from_str(dataset_case[label], boxes_dependencies=True)

    else:
        boxes = ast.literal_eval(dataset_case[boxes])
        label = dict_from_str(dataset_case[label])
    
    study_instance = dataset_case[study_instance]

    return {id: {'boxes': boxes, 'label': label, 'StudyInstanceUID': study_instance}}


def get_dict_study_level(case_study, columns):
    id = columns.index('id')
    NP = columns.index('Negative for Pneumonia')
    TA = columns.index('Typical Appearance')
    IA = columns.index('Indeterminate Appearance')
    AA = columns.index('Atypical Appearance')

    id = case_study[columns[id]].split('_')[0]
    NP = case_study[columns[NP]]
    TA = case_study[columns[TA]]
    IA = case_study[columns[IA]]
    AA = case_study[columns[AA]]

    return {id: {'Negative for Pneumonia': NP, 'Typical Appearance': TA, 'Indeterminate Appearance': IA, 'Atypical Appearance': AA}}


def dict_from_str(string: str, boxes_dependencies: bool =False) -> dict:
    elem_list = string.split(' ')
    opacities_idx = [i for i,x in enumerate(elem_list) if x=='opacity']
    count = 0
    label_dict = {}
    if not boxes_dependencies:
        for elem in elem_list:
            if elem=='opacity':
                name = 'opacity_' + str(count)
                if count == len(opacities_idx) - 1:
                    label_dict[name] = np.array([float(x) for x in elem_list[opacities_idx[count] + 1:]]).tolist()

                else:
                    label_dict[name] = np.array([float(x) for x in elem_list[opacities_idx[count] + 1:opacities_idx[count+1]]]).tolist()
                count += 1
    
    if boxes_dependencies:
        label = string.split(' ')
        label_dict[label[0]] = [float(x) for x in label[1:]]

    return label_dict


def create_dataset_json(train_image_study_path: str, 
                        train_study_level_path: str, 
                        save_path: str = r'C:\\Users\\MikołajStryja\\Documents\\Studia\\master\\dataset.json'):

    pd_til = data_loader.load_csv(train_image_study_path)
    pd_tsl = data_loader.load_csv(train_study_level_path)

    dataset = {}

    columns_til = list(pd_til.columns)
    # The train image level dataset is converted to the dictionary. 
    for elem in pd_til.iterrows(): 
        temp = get_dict_image_level(elem[1], columns_til)
        dataset.update(temp)

    dataset_tsl = {}

    columns_tsl = list(pd_tsl.columns)
    # The train study level dataset is converted to the dictionary.
    for elem in pd_tsl.iterrows():
        temp = get_dict_study_level(elem[1], columns_tsl)
        dataset_tsl.update(temp)

    # Based on the 'StudyInstanceUID' key, above dicts are connected in one. 
    suid = 'StudyInstanceUID'
    for key in dataset.keys():
        x = dataset[key][suid]
        dataset[key].update(dataset_tsl[x])

    files_and_dirctories.save_json(dataset, save_path)

    return dataset
    
