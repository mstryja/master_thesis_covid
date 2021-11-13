import numpy as np
from master_lib.utils import files_and_directories
import pandas as pd
from master_lib.model.generator import DataGenerator
from typing import List

import PIL
import os

# tensorflow & keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib


def dataset_partition(train_images_path_list: List, coef: float = 0.3):
    up = int(np.ceil(len(train_images_path_list)*(1 - coef)))
    return train_images_path_list[:up], train_images_path_list[up:]


def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


def main():
    dataset_path = r'C:\Users\MikołajStryja\Documents\Studia\master\dataset.json'
    dataset = files_and_directories.load_json(dataset_path)
    dataset_images = r'C:\Users\MikołajStryja\Documents\Studia\siim-covid19-detection'

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    print(device_lib.list_local_devices())
    num_classes = 4
    model_archive = r'C:\Users\MikołajStryja\Documents\Studia\master\trained_models'

    dataset = files_and_directories.load_json(dataset_path)
    train_imgs = files_and_directories.list_train_images(dataset_images)

    # callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=7, verbose=1)

    learning_rate = tf.keras.callbacks.LearningRateScheduler(
        scheduler, verbose=1
        )

    Xtrain, Xval = dataset_partition(train_imgs)
    training_generator = DataGenerator(Xtrain, batch_size=16)
    validation_generator = DataGenerator(Xval, batch_size=16)

    model4 = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(512, 512, 1)),
        layers.Conv2D(16, (5, 5), padding='same', activation='relu', name="first_conv2d_filter"),
        layers.MaxPooling2D(),
        layers.Conv2D(20, (3, 3), padding='same', activation='relu', name="second_conv2d_filter"),
        layers.MaxPooling2D(),
        layers.Conv2D(20, (3, 3), padding='same', activation='relu', name="third_conv2d_filter"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (5, 5), padding='same', activation='relu', name="fourth_conv2d_filter"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        # layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model4_name = 'Sequential_3'

    model4.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model4.fit(training_generator,
        epochs=20,
        validation_data=validation_generator,
        verbose=1,
        callbacks=[early_stop])

    model4.save(os.path.join(model_archive, model4_name))

    return 0

if __name__ == "__main__":
    main()

