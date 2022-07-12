
import numpy as np
from datetime import datetime 
import os
import csv
from project.environments.environment import Environment


from project.data_structures.replay_buffer import Replay_Buffer

from project.policies.q_value_function import Q_Value_Function

class DQN():
    def __init__(self,hyp,from_file = None):
        self.hyp = hyp
        self.env = Environment(hyp=self.hyp)
        self.ACTIONS, self.STATE_SHAPE = self.env.get_actions_and_obs_shape()
        self.policy = Q_Value_Function(hyp=self.hyp,a_size=self.ACTIONS,s_size=self.STATE_SHAPE)
        self.memory = Replay_Buffer(self.hyp["BUFFER_SIZE"],self.hyp["BATCH_SIZE"],self.hyp["STACK_SIZE"])


        self.stacked_state = None
        self.top_state = None

        base_dir = os.getcwd()
        self.log_dir = os.path.join(base_dir, "logs")

        self.state_dir = os.path.join(base_dir, "state_saves")
        self.log_path = None
        self.create_log()

        self.frame_i = 0
        self.ep_i = 0
        self.last_ep_rewards = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.last_ep_lengths = np.zeros(self.hyp["UPDATE_EVERY_N"])
        self.averaging_index = 0

        if from_file != None:
            self.read_log_state(from_file)

    def train(self):
        # learn until max number of frames reached
        frames_since_render = self.hyp["RENDER_EVERY_N"]
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_reward=0
            ep_length=0
            terminal = False
            self.stacked_state, self.top_state  = self.env.reset()
            action_i = 0
            while not terminal and action_i <= self.hyp["MAX_EP_ACTIONS"]:
                action = self.policy.choose_action(self.stacked_state,self.frame_i)
                new_stacked_state, reward, terminal, _, new_top_state = self.env.step(action)

                ep_reward += reward
                action_i += 1
                ep_length += 1

                # Experience in form [top_state,action,reward,new_top_state,term]
                self.memory.add_exp(self.stacked_state,action,reward,new_stacked_state,terminal)
                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                self.frame_i += 1
                
                # Replay from memory
                if  self.frame_i % self.hyp["REPLAY_FREQ"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    states,actions,rewards,new_states,terminals = self.memory.sample()
                    self.policy.learn(states,actions,rewards,new_states,terminals)
                
                # update target
                if  self.frame_i % self.hyp["UPDATE_TARGET"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    self.policy.update_target_model()

            self.last_ep_rewards[self.averaging_index] = ep_reward
            self.last_ep_lengths[self.averaging_index] = ep_length

            self.averaging_index = (self.averaging_index + 1) % self.hyp["UPDATE_EVERY_N"]

            if self.ep_i % self.hyp["RENDER_EVERY_N"] == 0:
                self.env.save_ep_gif(self.ep_i)
                self.env.save_ep_gif_processed(self.ep_i)

            if self.ep_i % self.hyp["UPDATE_EVERY_N"] == 0:
                av_reward = np.mean(self.last_ep_rewards)
                av_length = np.mean(self.last_ep_lengths)

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"{current_time} Ep: {self.ep_i},"
                + f" Frames: {self.frame_i},"
                + f" Av reward: {av_reward},"
                + f" Av ep length: {av_length},"
                + f" Epsilon: {self.policy.get_epsilon(self.frame_i)}")


            self.write_to_log(self.ep_i, self.frame_i, ep_reward, ep_length)

            if self.ep_i % self.hyp["SAVE_EVERY_N"] == 0:
                self.log_state() 

            self.ep_i += 1
            self.env.clear_ep_buffer()
    
    # creates log for ep length and rewards for graphing
    def create_log(self):
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        log_name = "./log" + "-" + str(current_time) +".csv"
        self.log_path = os.path.join(self.log_dir, log_name) 

    # writes ep length and reward in log
    def write_to_log(self, episode, frames, ep_reward, ep_frame_count):
        row = [frames, episode, ep_reward, ep_frame_count]
        with open(self.log_path, 'a',newline="\n") as out:
            csv_out = csv.writer(out, delimiter =',')
            csv_out.writerow(row)

    def log_state(self):
        # call to NN and RB as well
        # save frame_no count etc so can resume from where left off
        now = datetime.now()
        current_time = now.strftime("%d-%m-%y-%H:%M:%S")
        directory = os.path.join(self.state_dir, f"save-ep{self.ep_i}-{current_time}")
        os.mkdir(directory)

        self.policy.save_state(directory)
        self.memory.save_state(directory)
        self.save_state(directory)

    def read_log_state(self,directory):
        # load NN and RB as well
        self.policy.load_state(directory)
        self.memory.load_state(directory)
        self.load_state(directory)

    def save_state(self,directory):
        log_path = os.path.join(directory, "agent.csv")
        with open(log_path, 'a',newline="\n") as out:
            csv_out = csv.writer(out, delimiter =',')
            csv_out.writerow([self.log_path])
            csv_out.writerow([self.frame_i,self.ep_i,self.averaging_index])
            for i in range(self.hyp["UPDATE_EVERY_N"]):
                csv_out.writerow([self.last_ep_lengths[i],self.last_ep_rewards[i]])
        print(f"Successfully saved agent to {log_path}")

    def load_state(self,directory):
        log_path = os.path.join(directory, "agent.csv")
        f = open(log_path)
        reader = csv.reader(f, delimiter =',')
        i=0
        for row in reader:
            if i==0:
                self.frame_i, self.ep_i, self.averaging_index = int(row[0]), int(row[1]), int(row[2])
            if i==1:
                self.log_dir = row[0]
            else:
                self.last_ep_lengths[i-1], self.last_ep_rewards[i-1] = float(row[0]), float(row[1])
            i+=1
        print(f"Successfully loaded agent from {log_path}")



    def display_progress(self):
        raise(NotImplementedError)