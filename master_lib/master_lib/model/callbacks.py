import tensorflow as tf
import keras

class BCP(keras.callbacks.Callback):
    batch_accuracy = [] # accuracy at given batch
    batch_loss = [] # loss at given batch    
    def __init__(self):
        super(BCP,self).__init__() 
        
    def on_train_batch_end(self, batch, logs=None):                
        BCP.batch_accuracy.append(logs.get('accuracy'))
        BCP.batch_loss.append(logs.get('loss'))