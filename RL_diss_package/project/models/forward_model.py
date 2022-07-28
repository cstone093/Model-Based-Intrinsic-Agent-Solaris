import tensorflow as tf
import numpy as np

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

    def get_mean_stdev(self,error):
        self.n += 1.0
        e_sum, e_sq_sum, n = self.error_sum, self.error_sq_sum, self.n
        mu = (e_sum/n)
        sigma = np.sqrt(e_sq_sum/n - e_sum*e_sum/n/n)
        return mu, sigma

    def normalise_reward(self,error):
        self.error_sum += error
        self.error_sq_sum += error*error
        mu,sigma = self.get_mean_stdev(error)
        normalised = (error - mu)/sigma
        return normalised
