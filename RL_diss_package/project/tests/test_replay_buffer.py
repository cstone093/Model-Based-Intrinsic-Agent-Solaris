import pytest
import numpy as np

# from project.datastructures.Replay_Buffer import Replay_Buffer

# current_dir = os.getcwd()
# parent_dir = os.pardir
# rb_dir = os.path.join(parent_dir,"Data_Structures")

from project.data_structures.replay_buffer import Replay_Buffer
class TestRB:
    exp1 = ["s1","r",10,"s2"]
    exp2 = ["s2","r",9,"s3"]
    exp3 = ["s3","r",8,"s4"]


    ''' Tests for creation of buffer '''
    # Test that buffer can be instantiated
    def test_create_buffer(self):
        rb=Replay_Buffer(5,2)

    # Test that errors are thrown when batch is zero
    def test_exception_batch_size(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(5,0)

    # Test that errors are thrown when buffer is zero
    def test_exception_buffer_size(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(0,5)

    # Test that errors are thrown when batch size is greater than buffer size
    def test_exception_batch_buffer_incompatible(self):
        with pytest.raises(AssertionError):
            rb=Replay_Buffer(2,3)

    
    ''' Tests for adding to buffer '''
    def test_add(self):
        rb=Replay_Buffer(5,2)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
    
    def test_can_add_beyond_capacity(self):
        rb=Replay_Buffer(2,1)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp2)
        rb.add_exp(self.exp3)

    ''' Tests for sampling '''
    # test that sampling from empty buffer throws error
    def test_sample_empty(self):
        rb=Replay_Buffer(2,1)
        # empty buffer
        with pytest.raises(Exception):
            rb.sample()

    def test_sample_1(self):
        rb=Replay_Buffer(2,1)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        sample = rb.sample()
        assert len(sample) == 1

        assert sample == [self.exp1]

    def test_sample_2(self):
        rb=Replay_Buffer(2,2)
        # empty buffer
        # only one item in buffer
        rb.add_exp(self.exp1)
        rb.add_exp(self.exp1)
        sample = rb.sample()
        assert len(sample) == 2

        assert sample == [self.exp1,self.exp1]

    ''' Functionality Tests'''

    def test_queue(self):
        rb=Replay_Buffer(1,1)
        rb.add_exp(self.exp1)
        assert rb.sample() == [self.exp1]
        rb.add_exp(self.exp2)
        assert rb.sample() == [self.exp2]