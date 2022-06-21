from tkinter.tix import Tree
from PIL import Image
import numpy as np
import gym
from project.environments.environment import Environment


class Solaris(Environment):

    def __init__(
        self,
        env_name="ALE/Solaris-v5",
        stack_size=4,
        do_rescale = True,
        rescale_dims=(84,84),
        evaluation=False,
        do_crop=False,
        crop_borders=(0,0,0,0)
    ) -> None:
        self.env = gym.make(env_name, render_mode="human" if evaluation else None)
        self.state = None
        self.do_crop = do_crop
        self.crop_borders = crop_borders
        self.stack_size = stack_size
        self.do_rescale = do_rescale
        self.rescale_dims = rescale_dims
        self.observation_space = (self.rescale_dims[0], self.rescale_dims[1],self.stack_size)
        self.last_lives = 3
        self.reset_done = False

    def process_state(self, state):
        # Turn the image to grayscale and crop to active playing area, further resizing the size
        # to reduce computation resources whilst mantaining similar data
        # Converts the input of (210, 160, 3) to (84, 84, 1)
        #TODO look into whether solaris should be cropped
        grayscaled = Image.fromarray(state).convert("L").resize(self.rescale_dims)
        return np.array(grayscaled, dtype=np.uint8)[..., np.newaxis]

    def reset(self):
        start_state = self.env.reset()
        self.reset_done = True
        processed = self.process_state(start_state)
        self.state = np.repeat(processed, self.stack_size, axis=2)
        return np.array(self.state)

    def step(self, action):
        assert self.reset_done == True, "Environment must be reset before it can be stepped"
        new_state, reward, terminal, metadata = self.env.step(action)
        # print(f"step taken returned {reward},{terminal},{metadata}")
        life_lost = True if metadata["lives"] < self.last_lives else False
        if metadata["lives"] == 0:
            terminal = True
        self.last_lives = metadata["lives"]

        processed = self.process_state(new_state)

        # for stack_size 3
        self.state = np.append(self.state[:, :, 1:], processed, axis=2)

        # return frame stacked states, reward earned, whether the current state is terminal,
        # whether the agent lost a life, and the non-processes new state
        return np.array(self.state), reward, terminal, life_lost, new_state

    def get_actions_and_obs_shape(self):
        return self.env.action_space.n, self.observation_space