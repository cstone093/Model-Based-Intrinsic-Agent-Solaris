import pytest
import numpy as np

from project.data_structures.replay_buffer import Replay_Buffer
from project.environments.solaris import Solaris
class TestRB:
    env = Solaris()
        
    stacked_state, top_state = env.reset()
    a,s = env.get_actions_and_obs_shape()

    # curr_stacked_state, reward, terminal, life_lost, curr_top_state, unprocessed_new_state
    a = 1
    _ , reward, term, _ ,  new_top_state, _ = env.step(a)
    exp1 = [top_state,a,reward,new_top_state,term]

    top_state = new_top_state

    a = 2
    _ , reward, term, _ ,  new_top_state, _= env.step(a)
    exp2 = [top_state,a,reward,new_top_state,term]

    top_state = new_top_state

    a = 3
    _ , reward, term, _ ,  new_top_state, _ = env.step(a)
    exp3 = [top_state,a,reward,new_top_state,term]


    a = 4
    _ , reward, term, _ ,  new_top_state, _ = env.step(a)
    exp4 = [top_state,a,reward,new_top_state,False]



    ''' Tests for creation of buffer '''
    # Test that buffer can be instantiated
    def test_create_buffer(self):
        rb=Replay_Buffer(5,2,1)

    # Test that errors are thrown when batch is zero
    def test_exception_batch_size(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(5,0,1)

    # Test that errors are thrown when buffer is zero
    def test_exception_buffer_size(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(0,5,1)

    # Test that errors are thrown when batch size is greater than buffer size
    def test_exception_batch_buffer_incompatible(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(2,3,1)

    def test_exception_buffer_stack_incompatible(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(2,1,3)

    
    ''' Tests for adding to buffer '''
    def test_add(self):
        rb=Replay_Buffer(5,2,1)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
    
    def test_can_add_beyond_capacity(self):
        rb=Replay_Buffer(2,1,1)
        # empty buffer
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp2)
        # capacity full here
        rb.add_exp(self.exp3)

    ''' Tests for sampling '''
    # test that sampling from empty buffer throws error
    def test_sample_empty(self):
        rb=Replay_Buffer(2,1,1)
        # empty buffer
        with pytest.raises(Exception):
            rb.sample()

    def sample_hyp(self,buffer,batch,stack,exp):
        rb=Replay_Buffer(buffer,batch,stack)
        for i in range(exp):
            rb.add_exp(self.exp1)
        states,actions,rewards,new_states,terminals = rb.sample()
        assert states.shape == (batch, self.s[0], self.s[1], stack)
        assert actions.shape[0] == batch
        assert rewards.shape[0] == batch
        assert new_states.shape == (batch, self.s[0], self.s[1], stack)
        assert terminals.shape[0] == batch


    def test_sample(self):
        self.sample_hyp(10,3,2,8)

    # Test that can add and sample beyond size of buffer
    def test_sample_1(self):
        self.sample_hyp(10,3,2,13)

    # Test stack size 1
    def test_sample_2(self):
        self.sample_hyp(10,3,1,8)
    
    # Test stack size 0
    def test_sample_3(self):
        with pytest.raises(AssertionError):
            self.sample_hyp(10,3,0,8)

    # Test batch size 0
    def test_sample_4(self):
        with pytest.raises(AssertionError):
            self.sample_hyp(10,0,2,8)