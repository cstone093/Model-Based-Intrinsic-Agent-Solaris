
import pytest
import numpy as np

from project.environments.solaris import Solaris

class TestEnv:

    # Environment init
    def test_env_init(self):
        env = Solaris()

    def test_env_obs(self):
        r = (84,84)
        stack = 4
        env = Solaris(rescale_dims=r,stack_size=stack)
        
        a,o = env.get_actions_and_obs_shape()
        assert a == 18
        assert o == (r[0],r[1],stack)

    def test_reset(self):
        r = (84,84)
        stack = 4
        env = Solaris(rescale_dims=r,stack_size=stack)
        
        a,o = env.get_actions_and_obs_shape()

        stacked_state, top_state = env.reset()
        assert stacked_state.shape == o
        assert top_state.shape == (r[0],r[1],1)

    
    def test_no_step_before_reset(self):
        r = (84,84)
        stack = 4
        env = Solaris(rescale_dims=r,stack_size=stack)
        with pytest.raises(AssertionError):
            env.step(1)
    
    def test_step(self):
        r = (84,84)
        stack = 4
        env = Solaris(rescale_dims=r,stack_size=stack)
        
        a,o = env.get_actions_and_obs_shape()

        stacked_state, top_state = env.reset()

        #  curr_stacked_state, reward, terminal, life_lost, curr_top_state, unprocessed_new_state
        stacked_state, reward, term, life_lost, top_state = env.step(1)

        assert stacked_state.shape == o
        assert isinstance(reward,float)
        assert isinstance(term,bool)
        assert isinstance(life_lost,bool)
        assert top_state.shape == (r[0],r[1],1) 
