from collections import deque
import random
from re import I
import numpy as np


class Replay_Buffer():
  def __init__(self,buffer_size,batch_size,stack_size):
    assert batch_size > 0, "Batch size must be greater than zero"
    assert buffer_size > 0, "Buffer size must be greater than zero"
    assert stack_size > 0, "Stack size must be greater than zero"
    assert batch_size <= buffer_size, "Batch size must be smaller than buffer size"
    assert stack_size <= buffer_size, "Stack size must be smaller than buffer size"
    # self._buffer = deque(maxlen=buffer_size)
    self.BATCH_SIZE = batch_size
    self.STACK_SIZE = stack_size
    self.BUFFER_SIZE = buffer_size

    self._states = np.empty((self.BUFFER_SIZE,84,84),dtype=np.uint8)
    self._rewards = np.empty(self.BUFFER_SIZE,dtype=np.float32)
    self._actions = np.empty(self.BUFFER_SIZE,dtype=np.int8)
    self._terminals = np.empty(self.BUFFER_SIZE,dtype=np.bool8)

    self._exp_count = 0
    self._new_index = 0

  # adds new experiences into the buffer from the format
  # [st,at,rt,stprime,term]
  # to separate buffers
  def add_exp(self,exp):
    st,at,rt,stprime,term = exp
    self._states[self._new_index, ...] = st.squeeze()
    self._actions[self._new_index] = at
    self._rewards[self._new_index] = rt
    self._terminals[self._new_index] = term

    self._exp_count = np.max([self._exp_count, self._new_index + 1])
    self._new_index = (self._new_index + 1) % self.BUFFER_SIZE


  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):

    if self._exp_count < self.STACK_SIZE or self._exp_count == 0:
      raise Exception("Sufficient experience must be added before it is sampled")
    
    else:
      index_options = np.arange(self.STACK_SIZE,self._exp_count)
      index_options = index_options[
            (index_options < self._new_index - 1)
            | (index_options - self.STACK_SIZE > self._new_index - 1)]
      if len(index_options)==0:
        raise Exception("Sufficient experience must be added before it is sampled") # Need to handle this
      else:
        states_stacked = []
        for frame_i in np.arange(self.STACK_SIZE, 0, -1):
            states_stacked.append(index_options - frame_i)

        # removes indexes where the stacked states contain a terminal state
        index_options = index_options[
            np.logical_not(self._terminals[np.stack(states_stacked, axis=1)].any(axis=1))
        ]
        if len(index_options)==0:
          raise Exception("Sufficient experience must be added before it is sampled") # Need to handle this
        # select randomly, a state from the processed list of  options
        chosen_exp = np.random.choice(index_options, self.BATCH_SIZE)

        ss = np.array(
            [self._states[exp - self.STACK_SIZE : exp, ...] for exp in chosen_exp]
        )
        s_primes = np.array(
            [self._states[exp - self.STACK_SIZE + 1 : exp + 1, ...] for exp in chosen_exp]
        )

        # Transpose states from (batch_size, stack_size, 84, 84) to (batch_size, 84, 84, stack_size)
        return (
            np.transpose(ss, axes=(0, 2, 3, 1)),
            self._actions[chosen_exp],
            self._rewards[chosen_exp],
            np.transpose(s_primes, axes=(0, 2, 3, 1)),
            self._terminals[chosen_exp],
        )
    
  def save_state(self):
    raise(NotImplementedError)


  def load_state(self):
    raise(NotImplementedError)