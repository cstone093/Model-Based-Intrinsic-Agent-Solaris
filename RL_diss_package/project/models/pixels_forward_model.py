from project.models.forward_model import ForwardModel
import tensorflow as tf
import numpy as np

from keras import Model
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Conv2D, Dropout, MaxPooling2D, Flatten, Rescaling, Reshape, concatenate
from keras.layers import Input
from keras.optimizers import adam_v2, rmsprop_v2

from PIL import Image

class FM_Pixels(ForwardModel):

    def _create_CNN(self):
        initializer = tf.keras.initializers.HeNormal

        # Convolutional branch with the state

        state_input = Input(shape=self.S_SIZE)
        state_branch = Rescaling(scale=1.0/255)(state_input)
        state_branch = Conv2D(32,(8, 8),strides=4,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)
        state_branch = Conv2D(64,(4, 4),strides=2,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)
        state_branch = Conv2D(64,(3, 3),strides=1,kernel_initializer=initializer)(state_branch)
        state_branch = Activation("relu")(state_branch)
        state_branch = Flatten()(state_branch)
        state_branch = Model(inputs=state_input,outputs=state_branch)

        # Chosen to be class as no linearity of action representation
        action_input = Input(shape=self.A_SIZE)
        action_branch = Dense(self.A_SIZE)(action_input)
        action_branch = Model(inputs=action_input,outputs=action_branch)

        # Combine with action
        s_a_pair = concatenate([state_branch.output,action_branch.output])

        encoder = Dense(512,kernel_initializer=initializer,activation="relu")(s_a_pair)
        encoder = Dense(512,kernel_initializer=initializer,activation="relu")(encoder)
        encoder = Dense(np.prod(self.S_SIZE),kernel_initializer=initializer,activation="relu")(encoder)
        encoder = Reshape(self.S_SIZE)(encoder)

        model = Model(inputs=[state_input,action_input],outputs=encoder)

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=adam_v2.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )
        return model
    
    def learn(self,obs,acs,new_obs):
        training_in = zip(obs,acs)
        # Fit the model using our corrected values.
        self.model.fit(
            training_in,
            new_obs,
            batch_size=self.hyp["BATCH_SIZE"],
            verbose=0,
            shuffle=False,
        )

    def get_error(self,s,a,new_s):
        actions = np.zeros(shape=(1,self.A_SIZE),dtype=np.uint8)
        actions[0][a] = 1
        print(actions)
        s = np.array([s],dtype=np.float32)
        # sample = np.column_stack((a,*s))
        # # print(len(sample))
        # print(sample)
        # print(sample.shape, sample)
        # prediction = self.model.predict(sample)
        prediction = self.model.predict([s,actions],verbose=0)
        # im = Image.fromarray(prediction[:,:,0])
        # im.save("prediction.jpeg")
        L2_error = np.linalg.norm(prediction-new_s)
        print(f"L2 error is: {L2_error}")

        return L2_error

    # Need to do some scaling here
    def pixels_reward(self,s,a,new_s):
        error = self.get_error(s,a,new_s)
        return(error)
