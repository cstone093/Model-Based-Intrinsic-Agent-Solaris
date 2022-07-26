
from abc import abstractmethod
import numpy as np
from datetime import datetime 
from project.agents.dqn_agent import DQN

from project.models.pixels_forward_model import FM_Pixels

class Intrinsic_DQN(DQN):
    def __init__(self,hyp,from_file = None,alpha=1,beta=0,i_type="pixels"):
        super(Intrinsic_DQN,self).__init__(hyp=hyp,from_file=from_file)
        self.alpha = alpha
        self.beta = beta
        self.intrinsic_type = i_type
        self.forward_model = FM_Pixels(hyp,self.ACTIONS,self.STATE_SHAPE)

    def train(self):
        # learn until max number of frames reached
        while self.frame_i < self.hyp["EPS_FRAMES_FINAL"]:
            ep_reward=0
            ep_length=0
            terminal = False
            self.stacked_state, self.top_state  = self.env.reset()
            action_i = 0
            while not terminal and action_i <= self.hyp["MAX_EP_ACTIONS"]:
                action = self.policy.choose_action(self.stacked_state,self.frame_i)
                new_stacked_state, extr_reward, terminal, _, new_top_state = self.env.step(action)

                ep_reward += extr_reward
                action_i += 1
                ep_length += 1

                if self.intrinsic_type == "pixels":
                    # Do I do this with stacked?
                    intr_reward = self.forward_model.pixels_reward(self.stacked_state,action,new_stacked_state)
                    reward = self.alpha * extr_reward + self.beta * intr_reward
                else:
                    reward = extr_reward
                # Experience in form [top_state,action,reward,new_top_state,term]
                self.memory.add_exp(self.stacked_state,action,reward,new_stacked_state,terminal)  
                
                self.top_state = new_top_state
                self.stacked_state = new_stacked_state

                self.frame_i += 1
                
                # Replay from memory
                if  self.frame_i % self.hyp["REPLAY_FREQ"] == 0 and self.frame_i > self.hyp["EPS_FRAMES_INIT"]:
                    states,actions,rewards,new_states,terminals = self.memory.sample()
                    self.policy.learn(states,actions,rewards,new_states,terminals)
                    self.forward_model.learn(states,actions,new_states)
                
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

            self.env.clear_ep_buffer()

            if self.ep_i % self.hyp["SAVE_EVERY_N"] == 0:
                self.ep_i += 1
                self.log_state() 
            else:
                self.ep_i += 1

class IntrinsicReward():
    @abstractmethod
    def pixels_reward(self,obs,new_obs):
        pass
    
    def vae_reward(self,obs,new_obs):
        pass

    # def get_features(self, x, reuse):
    #     nl = tf.nn.leaky_relu
    #     x_has_timesteps = (x.get_shape().ndims == 5)
    #     if x_has_timesteps:
    #         sh = tf.shape(x)
    #         x = flatten_two_dims(x)
    #     with tf.variable_scope(self.scope + "_features", reuse=reuse):
    #         x = (tf.to_float(x) - self.ob_mean) / self.ob_std
    #         x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
    #     if x_has_timesteps:
    #         x = unflatten_first_dim(x, sh)
    #     return x