from random import randint, random
import numpy as np
from datetime import datetime 
import os
import csv
from project.environments.environment import Environment


from project.data_structures.replay_buffer import Replay_Buffer

# from project.hyperparameters.dqn_hyp import solaris_hyp
# from project.hyperparameters.test_hyp import test_hyp

from project.policies.q_value_function import Q_Value_Function

class DQN():
    def __init__(self,hyp):
        self.hyp = hyp
        self.env = Environment(hyp=self.hyp)
        self.ACTIONS, self.STATE_SHAPE = self.env.get_actions_and_obs_shape()
        self.policy = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
        self.memory = Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])
        self.frame_i = 0
        self.ep_i = 0
        self.stacked_state = None
        self.top_state = None

        self.all_ep_rewards = []
        self.all_ep_lengths = []

        self.create_log()

    def train(self):
        # learn until max number of frames reached
        eps_since_update = 0
        frames_since_render = 0
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_reward=0
            ep_length=0
            self.ep_i += 1
            terminal = False
            self.stacked_state, self.top_state  = self.env.reset()
            action_i = 0
            while not terminal and action_i < self.hyp["MAX_EP_ACTIONS"]:
                action = self.policy.choose_action(self.stacked_state)
                new_stacked_state, reward, terminal, _, new_top_state = self.env.step(action)

                ep_reward += reward
                action_i += 1
                ep_length += 1

                # Experience in form [top_state,action,reward,new_top_state,term]
                self.memory.add_exp([self.top_state,action,self.clip_reward(reward),new_top_state,terminal])

                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                self.frame_i += 1
                frames_since_render += 1
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
                    print(f"Ep: {self.ep_i}, Frames: {self.frame_i}, Av reward: {np.sum(self.all_ep_rewards[-eps_since_update:])/eps_since_update}, Av ep length: {np.sum(self.all_ep_lengths[-eps_since_update:])/eps_since_update}")
                    eps_since_update = 0
                
            
            if frames_since_render > self.hyp["RENDER_EVERY_N"]:
                frames_since_render = 0
                self.env.save_ep_gif(self.ep_i)

            eps_since_update += 1
            self.all_ep_lengths.append(ep_length)
            self.all_ep_rewards.append(ep_reward)

            self.write_to_log(self.ep_i, self.frame_i, ep_reward, ep_length)

            self.env.clear_ep_buffer()

    def save_state(self):
        # call for NN and RB to save their state as well
        raise(NotImplementedError)
    
    def clip_reward(self,reward):
        reward = max(reward,-1)
        reward = min(reward,1)
    
    def create_lr_graph(self):
        raise(NotImplementedError)

    # creates log for ep length and rewards for graphing
    def create_log(self):
        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        name = "./log" + "-" + str(current_time) +".csv"
        path = os.path.join(self.log_dir, name) 
        
        self.f = open(path, "w")
        self.log = csv.writer(self.f)

    # writes ep length and reward in log
    def write_to_log(self, episode, frames, ep_reward, ep_frame_count):
        row = [frames, episode, ep_reward, ep_frame_count]
        self.log.writerow(row)

    def create_eps_graph(self):
        raise(NotImplementedError)

    def display_progress(self):
        raise(NotImplementedError)

    def save_state(self):
        # call to NN and RB as well
        # save frame_no count etc so can resume from where left off
        # AND SAVE OWN STATE
        self.policy.save_state()
        self.env.save_state()
        self.memory.save_state()
        raise(NotImplementedError)

    def load_state(self):
        # load to NN and RB as well
        # set frame_no count etc
        raise(NotImplementedError)
        