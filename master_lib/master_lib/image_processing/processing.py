import cv2
import os
import numpy as np
import pydicom as pdcm
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd
from PIL import Image
from typing import List, Tuple
import random
# from master_lib.loader.data_loader import DATASET_JSON

def generate_numpy_dataset(file_list: List, dataset_dict: dict, division_coef: float = 0.2, do_resize: bool = True, max_range: int = 400):
    """
    Docstring must be written
    """
    train_cases = int(np.ceil(len(file_list[:max_range])*(1 - division_coef)))
    test_cases = max_range - train_cases
    train_images = file_list[:train_cases]
    test_images = file_list[train_cases:max_range]
    random.shuffle(train_images)

    print(len(test_images), len(train_images))

    log = 'Found a ' + str(train_cases) + ' images to train. Some of them: \n' + str(train_images[:5])
    print(log)
    if do_resize:
        X = np.zeros(shape=(train_cases, 512, 512))
        Y = np.zeros(shape=(train_cases, 4))
        X_val = np.zeros(shape=(test_cases, 512, 512))
        Y_val = np.zeros(shape=(test_cases, 4))

    for i, case in enumerate(train_images):
        img = load_dicom_image(case)
        if do_resize:
            img = resize(img, 512)
            X[i, :, :] = img
            case_metadata = dataset_dict[os.path.basename(case)]
            temp_y = np.array([case_metadata["Atypical Appearance"], case_metadata["Indeterminate Appearance"], case_metadata["Negative for Pneumonia"], case_metadata["Typical Appearance"]])
            Y[i, :] = temp_y

    print('Train set prepared!')
    log = 'Found a ' + str(test_cases) + ' images to train. Some of them: \n' + str(test_images[:5])
    print(log)

    for i, case in enumerate(test_images):
        img = load_dicom_image(case)
        if do_resize:
            img = resize(img, 512)
            X_val[i, :, :] = img
            case_metadata = dataset_dict[os.path.basename(case)]
            temp_y = np.array([case_metadata["Atypical Appearance"], case_metadata["Indeterminate Appearance"], case_metadata["Negative for Pneumonia"], case_metadata["Typical Appearance"]])
            Y_val[i, :] = temp_y

    return X, Y, X_val, Y_val


def load_data_as_numpy_arr(path: str, do_resize: bool = True):
    img = load_dicom_image(path)
    if do_resize:
        img = resize(img, 512)

    return np.asarray(img)
    

def load_dicom_image(path: str):
    """
    Return dicom as the numpy array formated to the <0, 255> limits.
    """
    return dicom_to_array(path)


def dicom_to_array(path, voi_lut=True, fix_monochrome=True):
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


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS, return_numpy: bool = False):
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    if return_numpy:
        return np.asarray(im)

    return im


def get_bbox(path: str):
    """
    Returns the bboxes list from the DATASET_JSON file.
    Args: 
        path: str => path to the dicom image.
    """
    key = os.path.basename(path)
    bboxes = DATASET_JSON[key]['boxes']
    return bboxes


def get_bboxes_shape(path: str):
    """
    Returns bboxes of given image in the numpy array format: 
    (x, y, h, w)

    Args: 
        path: str => path to the dicom image.
    """
    bboxes = get_bbox(path)
    bboxes_arr = []
    for box in bboxes:
        x, y = int(np.ceil(box['x'])), int(np.ceil(box['y']))
        h, w = int(np.ceil(box['height'])), int(np.ceil(box['width']))
        bboxes_arr.append(np.array([x, y, h, w]))

    return np.asarray(bboxes_arr)


def get_bbox_ranges(bbox: dict):
    # TO CHANGE
    """
    Docstring must be extended. 
    Function returns the Tuple object of bounding box coordinates as int types. 
    Following order is kept: x1, x2, y1, y2. 
    Apply on image: 
    img[result[1]:result[2], result[3]:result[4]]
    """
    x1, y1 = int(np.ceil(bbox['x'])), int(np.ceil(bbox['y']))
    x2, y2 = x1 + int(np.ceil(bbox['width'])), y1 + int(np.ceil(bbox['height']))
    return (x1, x2, y1, y2)


def apply_bbox_from_metadata(img, bounding_box: List):
    # TO CHANGE
    for box in bounding_box:
        point_1, point_2 = reactangle_from_bbox(box)
        img = cv2.rectangle(img, point_1, point_2, color=(0, 255, 255), thickness=3)
    return img


def reactangle_from_bbox(bounding_box: dict):
    # TO CHANGE
    """
    Create reactangle in points interpretation from bounding_box dictionary, located in metadata.

    Params: 
        bounding_box: dict -> dictionary with bounding box parameters in format complies with
                                dataset.json file. 

    Returns:
        (point_1, point_2) -> tuple of two points, each point is represented by two int values: 
                                (x, y) and returned as tuple.
    """
    x1, y1 = int(np.ceil(bounding_box['x'])), int(np.ceil(bounding_box['y']))
    x2, y2 = x1 + int(np.ceil(bounding_box['width'])), y1 + int(np.ceil(bounding_box['height']))
    return (x1, y1), (x2, y2)


def reactangle_from_list(bounding_box: np.ndarray):
    """

    """
    for box in bounding_box:
        # point_1 = Point(int(np.ceil(box[1])), int(np.ceil(box[2])))
        # point_2 = point_1 + Point(int(np.ceil(box[3])), int(np.ceil(box[4])))
        x1, y1 = int(np.ceil(box[1])), int(np.ceil(box[2]))
        x2, y2 = x1 + int(np.ceil(box[3])), y1 + int(np.ceil(box[4]))

    return (x1, y1), (x2, y2)


def cropp_image_bbox(img: np.ndarray, bbox: np.ndarray):
    x1, y1 = int(np.ceil(bbox[1])), int(np.ceil(bbox[2]))
    x2, y2 = x1 + int(np.ceil(bbox[3])), y1 + int(np.ceil(bbox[4]))

    cropp = img[x1:x2, y1:y2]

    return cropp


def cropp_image_bbox_dict(img: np.ndarray, bbox: dict):
    x1, y1 = int(np.ceil(bbox['x'])), int(np.ceil(bbox['y']))
    x2, y2 = x1 + int(np.ceil(bbox['width'])), y1 + int(np.ceil(bbox['height']))

    cropp = img[y1:y2, x1:x2]

    return cropp


def change_bbox_size(path: str, img_orig_shape: Tuple[int, ...], new_shape=512):
    """

    """
    bboxes = get_bboxes_shape(path)
    new_boxes = []
    for box in bboxes:
        # (x, y, h, w)
        # img shape : (y, x)
        y_coef = box[1]/img_orig_shape[0]
        x_coef = box[0]/img_orig_shape[1]
        h_coef = box[2]/img_orig_shape[0]
        w_coef = box[3]/img_orig_shape[1]
        new_boxes.append(np.array([x_coef*new_shape, y_coef*new_shape, h_coef*new_shape, w_coef*new_shape]))

    return np.asarray(new_boxes)
