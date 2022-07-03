# from PIL import Image
# import numpy as np
# import gym
# from project.environments.environment import Environment

# class Test_Env(Environment):
#     def __init__(
#         self,
#         env_name="ALE/Pong",
#         stack_size=1,
#         do_rescale = False,
#         rescale_dims=(84,84),
#         evaluation=False,
#         do_crop=False,
#         crop_borders=(0,0,0,0)
#     ) -> None:
#         self.env = gym.make(env_name, render_mode="human" if evaluation else None)
#         self.curr_stacked_state = None
#         self.curr_top_state = None
#         self.do_crop = do_crop
#         self.crop_borders = crop_borders
#         self.stack_size = stack_size
#         self.do_rescale = do_rescale
#         self.rescale_dims = rescale_dims
#         self.observation_space = (self.rescale_dims[0], self.rescale_dims[1],self.stack_size)
#         self.last_lives = 3
#         self.reset_done = False