import PIL
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.utils import Sequence

from master_lib.image_processing.processing import load_data_as_numpy_arr
from master_lib.utils.files_and_directories import load_json
from master_lib.loader.data_loader import evaluate_categories
from master_lib.image_processing.preprocessing import *
from os.path import basename



class DataGenerator(Sequence):
    def __init__(self, list_IDs: List, batch_size=32, dim=(512, 512), n_channels=1, n_classes=4, shuffle=True, equalization: int = 1):
        """
        Constructor of DataGenerator class. 

        Args:
            list_IDS: List => List of train images path
            batch_size: int => number of image loaded into one batch
            dim: Tuple[int, int] => Input dimension of the image
            equalization: int => If 1 - traditional equalization, If 2 - adaptive equalization, If 0 - No equalization.

        """
        self.batch_size = batch_size
        self.dim = dim
        # self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.equalization = equalization
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            img = load_data_as_numpy_arr(ID)
            if self.equalization == 0 or self.equalization > 2: 
                X[i,] = img
            
            elif self.equalization == 1:
                X[i,] = histogram_equalization(img)

            elif self.equalization == 2:
                X[i,] = adaptive_equalization(img)

            # Store class
            y[i] = evaluate_categories(basename(ID))

        return X, y
    

class DataPatchGenerator(Sequence):
    def __init__(self, list_IDs, dataset_path, batch_size, patch_size, input_dim, channels=1, output_classes=4, shuffle=True, step_size=0.8):
        self.list_IDs = list_IDs
        self.datataset = load_json(dataset_path)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.channels = channels
        self.output_classes = output_classes
        self.shuffle = shuffle
        self.step_size = step_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.patch_size))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            patches = convert_image_to_patches(load_data_as_numpy_arr(ID, do_resize=False), self.patch_size, self.step_size)
            for patch in patches:
                # Here something should be changed if patches are returned.
                pass
            X[i,] = load_data_as_numpy_arr(ID)

            # Store class
            y[i] = evaluate_categories(basename(ID))

        return X, y
    
