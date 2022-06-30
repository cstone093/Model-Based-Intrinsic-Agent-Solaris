
from project.environments.solaris import Solaris
from project.environments.test_env import Test_Env


from project.data_structures.replay_buffer import Replay_Buffer

from project.hyperparameters.dqn_hyp import solaris_hyp
from project.hyperparameters.test_hyp import test_hyp

from project.policies.q_value_function import Q_Value_Function

class DQN():
    def __init__(self,solaris):
        if solaris:
            self.hyp = solaris_hyp
            # NEED: Q_network, env, RB
            self.env = Solaris(stack_size=self.hyp["STACK_SIZE"])
        else:
            self.hyp = test_hyp
            self.env = Test_Env(stack_size=self.hyp["STACK_SIZE"])
        self.ACTIONS, self.STATE_SHAPE = self.env.get_actions_and_obs_shape()
        self.q_network = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
        self.memory = Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])

    def train(self):
        # NEED TO HANDLE EXCEPTIONS
        print("WORKS")

    def save_gif(self):
        raise(NotImplementedError)

    def save_state(self):
        # call for NN and RB to save their state as well
        raise(NotImplementedError)

    def display_progress(self):
        raise(NotImplementedError)

    def save_state(self):
        # call to NN and RB as well
        raise(NotImplementedError)

    def load_state(self):
        # load to NN and RB as well
        raise(NotImplementedError)
