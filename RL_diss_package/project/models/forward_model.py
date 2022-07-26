import tensorflow as tf
import numpy as np

from keras import Model
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, Rescaling, Reshape, concatenate
from keras.layers import Input
from keras.optimizers import adam_v2, rmsprop_v2

class ForwardModel():
    def __init__(self,hyp,a_size,s_size):
        self.hyp = hyp
        self.A_SIZE = a_size
        self.S_SIZE = s_size
        self.learning_rate = hyp["INIT_LEARNING_RATE"]
        np.random.seed(self.hyp["SEED"])

        self.model = self._create_CNN()

        print("Forward Model Initialised")
        self.model.summary()
        
        self.error_sum = 0
        self.error_sq_sum = 0
        self.n = 0

    def get_stdev(self,error):
        self.error_sum += error
        self.error_sq_sum += error*error
        self.n += 1.0
        e_sum, e_sq_sum, n = self.sum, self.sum2, self.n
        return np.sqrt(e_sq_sum/n - e_sum*e_sum/n/n)

    def get_mean(self,error):
        pass
