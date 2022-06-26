import pytest
import numpy as np

from project.data_structures.replay_buffer import Replay_Buffer
class TestRB:
    exp1 = ["s1",1,"r","s2",False]
    exp2 = ["s2",1,"r","s3",False]
    exp3 = ["s3",3,"r","s4",False]
    # state, action, reward, new state, terminal

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

    def test_sample_1(self):
        rb=Replay_Buffer(2,1,1)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        sample = rb.sample()
        assert len(sample) == 1

        assert sample == [[self.exp1]]

    def test_sample_2(self):
        rb=Replay_Buffer(2,2,1)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp1)
        sample = rb.sample()
        assert len(sample) == 2

        assert sample == [[self.exp1],[self.exp1]]

    def test_sample_3(self):
        rb=Replay_Buffer(10,1,2)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp1)
        sample = rb.sample()
        assert len(sample) == 1

        assert sample == [[self.exp1,self.exp1]]

    ''' Functionality Tests'''

    # test that adding experience pushes old from end of queue
    def test_queue(self):
        rb=Replay_Buffer(1,1,1)
        rb.add_exp(self.exp1)
        assert rb.sample() == [[self.exp1]]
        rb.add_exp(self.exp2)
        assert rb.sample() == [[self.exp2]]