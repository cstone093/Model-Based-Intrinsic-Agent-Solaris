from collections import deque
import random
from re import I
import numpy as np


class Replay_Buffer():
  def __init__(self,buffer_size,batch_size,stack_size):
    assert batch_size > 0, "Batch size must be greater than zero"
    assert buffer_size > 0, "Buffer size must be greater than zero"
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
    print("info:",st.shape,at,rt,term)
    print("Index is: ",self._new_index)
    self._states[self._new_index] = st.squeeze()
    self._actions[self._new_index] = at
    self._rewards[self._new_index] = rt
    self._terminals[self._new_index] = term

    self._exp_count = max(self._exp_count, self._new_index + 1)
    self._new_index = (self._new_index + 1) % self.BUFFER_SIZE


  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):

    if self._exp_count < self.STACK_SIZE or self._exp_count == 0:
      raise Exception("Sufficient experience must be added before it is sampled")
    
    else:
      sample_states = np.empty((self.BATCH_SIZE,self.STACK_SIZE,84,84),dtype=np.uint8)
      sample_actions = np.empty(self.BATCH_SIZE,dtype=np.int8)
      sample_rewards = np.empty(self.BATCH_SIZE,dtype=np.float32)
      sample_next_states = np.empty((self.BATCH_SIZE,self.STACK_SIZE,84,84),dtype=np.uint8)
      sample_terminals = np.empty(self.BATCH_SIZE,dtype=np.bool8)

      possible_ixs = np.arange(self.STACK_SIZE,self._exp_count)
      possible_ixs = possible_ixs[
            (possible_ixs < self._new_index - 1)
            | (possible_ixs - self.STACK_SIZE > self._new_index - 1)]
      print(possible_ixs)
      if len(possible_ixs)==0:
        raise Exception("Sufficient experience must be added before it is sampled")
      else:
        stacks = []
        for number in np.arange(self.STACK_SIZE, 0, -1):
            stacks.append(possible_ixs - number)

        possible_ixs = possible_ixs[
            np.logical_not(self._terminals[np.stack(stacks, axis=1)].any(axis=1))
        ]
        # Choose a random batch of indices
        indices = np.random.choice(possible_ixs, self.BATCH_SIZE)

        states = np.array(
            [self._states[index - self.STACK_SIZE : index, ...] for index in indices]
        )
        future_states = np.array(
            [self._states[index - self.STACK_SIZE + 1 : index + 1, ...] for index in indices]
        )

        # Transpose states from (batch_size, stack_size, 84, 84) to (batch_size, 84, 84, stack_size)
        return (
            np.transpose(states, axes=(0, 2, 3, 1)),
            self._actions[indices],
            self._rewards[indices],
            np.transpose(future_states, axes=(0, 2, 3, 1)),
            self._terminals[indices],
        )
    
    # Need to make the code for this much tidier