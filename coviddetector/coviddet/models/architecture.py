from unicodedata import name
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50, DenseNet121, Xception
from keras.layers import Input, Conv2D



class CovidClassifier(keras.Model):

    def __init__(self, n_classes: int = 4, hidden_dense: int = 3, **params):
        super(CovidClassifier, self).__init__()
        # Network parameters:
        self.hidden_layers = hidden_dense

        self.with_sequential = params['Sequential'] if params is not None else True

        # Model
        self.converter = Conv2D(3, (1, 1), trainable=False, name='Converter')
        self.resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                               input_shape=None, pooling=None, classes=n_classes)
        self.glob_ava_pooling_2d = layers.GlobalAveragePooling2D(name="GLobAvaPooling2D")
        self.initializer = keras.initializers.GlorotUniform(seed=42)
        self.activation =  keras.activations.softmax
        
        self.dense_shapes = self.calculate_depth()
        self.sequentional_block = Sequential(name="Sequential_Block")
        for i, s in enumerate(self.dense_shapes):
            name = 'DenseBlock_' + str(i)
            self.sequentional_block.add(layers.Dense(s, activation='relu', name=name))
            
        
        self.outputs = layers.Dense(n_classes,
                                    kernel_initializer=self.initializer,
                                    activation=self.activation,
                                    name="ModelOutput")
        
    def call(self, inputs):
        x = self.converter(inputs)
        x = self.resnet(x)
        x = self.glob_ava_pooling_2d(x)
        if self.with_sequential:
            x = self.sequentional_block(x)

        return self.outputs(x)  
    
    def get_config(self):
        config = super(CovidClassifier, self).get_config()
        
    def calculate_depth(self):
        out_shape = self.resnet.output.shape[-1]
        out_shape /= 2
        shapes = []
        while out_shape > 4:
            shapes.append(int(out_shape))
            out_shape /= 2
            print(out_shape)

        return shapes


class CovidClassifierInceptionNet(keras.Model):
    def __init__():
        pass
        