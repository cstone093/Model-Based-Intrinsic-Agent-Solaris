from PIL import Image
import numpy as np
import gym

class Environment:
    def __init__(
        self,
        env_name,
        stack_size=1,
        do_rescale = False,
        rescale_dims=(84,84),
        evaluation=False,
        do_crop=False,
        crop_borders=(0,0,0,0)
    ) -> None:
        assert not env_name == None, "An environment name must be specified"
        self.env = gym.make(env_name, render_mode="human" if evaluation else None)
        self.curr_stacked_state = None
        self.curr_top_state = None
        self.do_crop = do_crop
        self.crop_borders = crop_borders
        self.stack_size = stack_size
        self.do_rescale = do_rescale
        self.rescale_dims = rescale_dims
        self.observation_space = (self.rescale_dims[0], self.rescale_dims[1],self.stack_size)
        self.last_lives = 3
        self.reset_done = False

    def process_state(self, state):
        # Converts the RGB input of (210, 160, 3) to grayscale (84, 84, 1)
        grayscaled = Image.fromarray(state).convert("L").resize(self.rescale_dims)
        return np.array(grayscaled, dtype=np.uint8)[..., np.newaxis]

    def reset(self):
        start_state = self.env.reset()
        self.reset_done = True
        processed = self.process_state(start_state)
        self.curr_stacked_state = np.repeat(processed, self.stack_size, axis=2)
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

        # TODO store unprocessed for each episode so that a gif can be made

        # return frame stacked states, reward earned, whether the current state is terminal,
        # whether the agent lost a life, and the non-processes new state
        return np.array(self.curr_stacked_state), reward, terminal, life_lost, self.curr_top_state

    def get_actions_and_obs_shape(self):
        return self.env.action_space.n, self.observation_space

    def save_ep_gif(self):
        raise(NotImplementedError)