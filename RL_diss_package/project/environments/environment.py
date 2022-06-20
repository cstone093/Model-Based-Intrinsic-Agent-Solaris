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
        self.env = gym.make(env_name, render_mode="human" if evaluation else None)
        self.state = None
        self.do_crop = do_crop
        self.crop_borders = crop_borders
        self.stack_size = stack_size
        self.do_rescale = do_rescale
        self.rescale_dims = rescale_dims
        self.observation_space = (self.rescale_dims[0], self.rescale_dims[1],self.stack_size)

    def process_state(self, state):
        raise NotImplementedError("Not Implemented process_state()")

    def reset(self):
        raise NotImplementedError("Not Implemented reset()")

    def step(self, action):
        raise NotImplementedError("Not Implemented step()")

    def get_actions_and_obs_shape(self):
        raise NotImplementedError("Not Implemented get_actions_and_obs_shape()")