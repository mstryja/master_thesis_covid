"""
Imports must be added!
"""
import os
import numpy as np
import pydicom as pdcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from typing import List, Tuple
from coviddet.utils.file_and_directories import load_json, list_train_images


class Dataset:
    # CLASSES = ['Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia', 'StudyInstanceUID', 'Typical Appearance', 'boxes', 'label']
    CLASSES = ['Atypical Appearance', 'Indeterminate Appearance', 'Negative for Pneumonia','Typical Appearance']
    def __init__(self, files: List, json_descriptor_path: str, classes=None, augmentation=False): #, """augmentation=None, preprocessing=None, resize=None, gray=None"""):
        self.images = files
        self.dataset_desc = load_json(json_descriptor_path)
        self.labels = [self.evaluate_categories(os.path.basename(img)) for img in self.images]

        self.augmentation = augmentation
        
    def __getitem__(self, i):
        if isinstance(i, slice):
            indices = range(*i.indices(len(self.images)))
            return [self.get_id(i) for i in indices]
        
        return self.get_id(i)

    def get_id(self, i: int):
        img = self.dicom_to_array(self.images[i])
        label = self.labels[i]
        return img, label
        
    def __len__(self):
        return len(self.images)

    def dicom_to_array(self, path, voi_lut=True, fix_monochrome=True):
        dicom = pdcm.read_file(path)
        # VOI LUT (if available by DICOM device) is used to
        # transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        return data

    def augment_operations(self):
        statistics = self.analyse_dataset_labels()
        to_augment = np.where(statistics < statistics.max())
        no_augment = np.where(statistics == statistics.max())
        nums_of_augmentation = []
        for i in to_augment:
            nums_of_augmentation.append(np.floor(statistics.max()/statistics[i]))

        nums_of_augmentation.insert(no_augment, 0)

        return nums_of_augmentation

    def evaluate_categories(self, ID: str):
        """
        Docstring must be written
        """
        evaluated_case = self.dataset_desc[ID]
        categories = []
        for col in self.CLASSES:
            categories.append(evaluated_case[col])

        return np.array(categories)

    def analyse_dataset_labels(self):
        statistics = np.zeros(shape=self.labels[0].shape)
        for label in self.labels:
            statistics += label

        return statistics

    def analyse_dataset_shapes(self):
        statistics = []
        for img in self.images:
            statistics.append(self.dicom_to_array(img).shape)

        return statistics


    @staticmethod
    def dataset_partition(train_images_path_list: List, coef: float = 0.3, evaluation_cases: int = 150, verbose: int = 1):
        max_idx = len(train_images_path_list)-evaluation_cases
        up = int(np.ceil(max_idx*(1 - coef)))
        if verbose==1:
            o = f"dataset length: {len(train_images_path_list)}\nEvaluation Cases: {evaluation_cases}"
            o += f"\nTrain cases: {up}\nValidation cases: {max_idx - up}"
            print(o)
        return train_images_path_list[:up], train_images_path_list[up:max_idx], train_images_path_list[max_idx:]

    @staticmethod
    def create_datasets(dataset_path: List, json_path: str, coef: float, test_cases: int, verbose: int = 1):
        files = list_train_images(dataset_path)
        train_set, val_set, test_set = Dataset.dataset_partition(files, coef, test_cases, verbose=verbose)
        return Dataset(train_set, json_path), Dataset(val_set, json_path), Dataset(test_set, json_path)
