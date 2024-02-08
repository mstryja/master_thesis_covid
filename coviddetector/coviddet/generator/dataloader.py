from coviddet.generator.dataset import Dataset
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Preprocessing and additionals:
from sklearn.preprocessing import MinMaxScaler
from sklearn import exposure
import albumentations as A
from albumentations import Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip, Rotate
from skimage import exposure
scaler = MinMaxScaler()

import cv2


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=8, input_shape=(1024, 1024), augmentation=None, preprocessing=None, shuffle=False, expanddims=False, **params):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.input_shape = input_shape
        self.indexes = np.arange(len(dataset))
        self.expand = expanddims

        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            image, label = self.dataset[j][0], self.dataset[j][1]
            image = cv2.resize(image, self.input_shape, interpolation = cv2.INTER_AREA)
                
            if self.preprocessing=='HEqualization':
                # Histogram equalization
                image = exposure.equalize_hist(image)
            elif self.preprocessing=='MinMax':
                # MinMax scaller
                scaler.fit(image)
                image = scaler.transform(image)
            else:
                pass
            
            if self.expand:
                image = np.expand_dims(image, axis=-1)
        
            data.append([image, label])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        # print(batch.shape)
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)