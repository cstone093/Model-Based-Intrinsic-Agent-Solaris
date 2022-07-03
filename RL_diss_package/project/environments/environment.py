from PIL import Image
import numpy as np
import gym
import os
import imageio

class Environment:
    def __init__(
        self,hyp
    ) -> None:
        self.hyp = hyp
        assert not self.hyp["ENV"] == None, "An environment name must be specified"
        self.env = gym.make(self.hyp["ENV"], render_mode="human" if self.hyp["EVALUATION"] else None)

        self.curr_stacked_state = None
        self.curr_top_state = None

        self.actions_space, self.observation_space = self.get_actions_and_obs_shape()

        self.last_lives = 3
        self.reset_done = False
        self.new_ep = True

        base_dir = os.getcwd()

        self.gif_dir = os.path.join(base_dir, "gif")

        self.ep_states = []

    def process_state(self, state):
        # Converts the RGB input of (210, 160, 3) to grayscale (84, 84, 1)
        if self.hyp["DO_RESCALE"]:
            state = Image.fromarray(state).convert("L").resize(self.hyp["RESCALE_DIMS"])
        return np.array(state, dtype=np.uint8)[..., np.newaxis]

    def reset(self):
        start_state = self.env.reset()
        self.reset_done = True
        processed = self.process_state(start_state)
        self.curr_stacked_state = np.repeat(processed, self.hyp["STACK_SIZE"], axis=2)
        self.clear_ep_buffer()
        return np.array(self.curr_stacked_state), processed

    # returns curr_stacked_state, reward, terminal, life_lost, curr_top_state, unprocessed_new_state
    def step(self, action):
        assert self.reset_done == True, "Environment must be reset before it can be stepped"
        unprocessed_new_state, reward, terminal, metadata = self.env.step(action)
        life_lost = True if metadata["lives"] < self.last_lives else False
        # if metadata["lives"] == 0:
        #     terminal = True
        self.last_lives = metadata["lives"]
        self.curr_top_state = self.process_state(unprocessed_new_state)
        self.curr_stacked_state = np.append(self.curr_stacked_state[:, :, 1:], self.curr_top_state, axis=2)
        
        # Store all states for an episode so a gif can be greated if called by the agent
        self.ep_states.append(unprocessed_new_state)

        # return frame stacked states, reward earned, whether the current state is terminal,
        # whether the agent lost a life, and the non-processes new state
        return np.array(self.curr_stacked_state), reward, terminal, life_lost, self.curr_top_state

    def get_actions_and_obs_shape(self):
        return self.env.action_space.n, (self.hyp["RESCALE_DIMS"][0],self.hyp["RESCALE_DIMS"][1],self.hyp["STACK_SIZE"])

    def clear_ep_buffer(self):
        self.ep_states = []

    def save_ep_gif(self,ep_no):
        # Create gifs from all saved sets of frames
        if np.shape(self.ep_states) == 0:
            print("No frames found!")
        else:
            name = "./gif" + "-" + str(ep_no) + ".gif"
            path = os.path.join(self.gif_dir, name)
            print(f"Length of episode was {len(self.ep_states)}")
            frames = [
                np.array(Image.fromarray(frame).resize((240, 315)), dtype=np.uint8)
                for frame in self.ep_states
            ]
            imageio.mimsave(
                path,
                frames,
                format="GIF",
                fps=60,
            )
            print(f"Episode {ep_no} gif created")

    def save_state(self):
        raise(NotImplementedError)