import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model

class Q_Value_Function:

    def __init__(self,hyp,a_size,s_size):
        self.hyp = hyp
        self.A_SIZE = a_size
        self.S_SIZE = s_size
        self.epsilon = hyp["EPSILON_START"]
        self.learning_rate = hyp["INIT_LEARNING_RATE"]
        self.local_model = self._create_CNN()
        self.target_model = self._create_CNN()

        weights = self.get_weights(self.local_model)
        self.set_weights(self.target_model,weights)
        # lr, loss, conv layers, dense, output size
        

    # Creates a CNN for the policy
    def _create_CNN(self):

        model = Sequential()
        model.add(InputLayer(input_shape=self.S_SIZE))

        model.add(Conv2D(32,(8, 8),strides=4,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),))
        model.add(Activation("relu"))

        model.add(Conv2D(64,(4, 4),strides=2,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),))
        model.add(Activation("relu"))

        model.add(Conv2D(64,(3, 3),strides=1,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)))
        model.add(Dense(self.A_SIZE, activation="linear"))

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model

    # Takes a batch of experience and performs back propagation on the CNN
    def learn(self,experience):

        current_states, actions, rewards, future_states, terminals = np.transpose(experience)

        # Get predicted Q values for each s using our current model
        curr_q_values = self.local_model.predict(current_states)
        # Get Q values for each new_s using the target model
        future_q = self.target_model.predict(future_states)

        updated_q_values = rewards + self.gamma * np.max(future_q, axis=1) * (
            1 - terminals
        )

        for index, (action, new_q) in enumerate(zip(actions, updated_q_values)):
            curr_q_values[index][action] = new_q

        # Fit the model using our corrected values.
        self.model.fit(
            current_states,
            curr_q_values,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )



    # Given a state, uses the CNN as a policy to choose an action
    def choose_action(self,state):
        # do epsilon soft
        if np.random.uniform(0,1) <= self.epsilon:
            return np.random.randint(self.A_SIZE)
        else:
            values = self.local_model.predict(state)
            return np.argmax(values)

    def set_weights(self,model,weights):
        model.set_weights(weights)

    def get_weights(self,model):
        weights = model.get_weights()
        return weights

