import numpy as np
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
        self.policy = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
        self.memory = Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])
        self.frame_i = 0
        self.ep_i = 0
        self.stacked_state = None
        self.top_state = None

    def train(self):
        # learn until max number of frames reached
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_rewards=[]
            self.ep_i += 1
            terminal = False
            self.stacked_state, self.top_state  = self.env.reset()
            action_i = 0
            while not terminal and action_i < self.hyp["MAX_EP_ACTIONS"]:
                action = self.policy.choose_action(self.stacked_state)
                new_stacked_state, reward, terminal, _, new_top_state = self.env.step(action)

                ep_rewards.append(reward)
                action_i += 1

                # Experience in form [top_state,action,reward,new_top_state,term]
                self.memory.add_exp([self.top_state,action,self.clip_reward(reward),new_top_state,terminal])

                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                self.frame_i += 1
                # Replay from memory
                if  self.frame_i % self.hyp["REPLAY_FREQ"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    try:
                        exp = self.memory.sample()
                        self.policy.learn(exp)
                    except:
                        pass
                
                # update target
                if  self.frame_i % self.hyp["UPDATE_TARGET"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    self.policy.update_target_model()

                if self.frame_i % self.hyp["UPDATE_EVERY_N"] == 0:
                    print(f"Ep: {self.ep_i}, Frames: {self.frame_i}, Av reward: {np.mean(ep_rewards)}")

    def save_state(self):
        # call for NN and RB to save their state as well
        raise(NotImplementedError)
    
    def clip_reward(self,reward):
        reward = max(reward,-1)
        reward = min(reward,1)

    def display_progress(self):
        raise(NotImplementedError)

    def save_state(self):
        # call to NN and RB as well
        # save frame_no count etc so can resume from where left off
        raise(NotImplementedError)

    def load_state(self):
        # load to NN and RB as well
        # set frame_no count etc
        raise(NotImplementedError)
