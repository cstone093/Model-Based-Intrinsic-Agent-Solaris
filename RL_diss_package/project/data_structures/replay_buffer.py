from collections import deque
import random

class Replay_Buffer():
  def __init__(self,buffer_size,batch_size,stack_size):
    assert batch_size > 0, "Batch size must be greater than zero"
    assert buffer_size > 0, "Buffer size must be greater than zero"
    assert batch_size <= buffer_size, "Batch size must be smaller than buffer size"
    self._buffer = deque(maxlen=buffer_size)
    self.BATCH_SIZE = batch_size
    self.STACK_SIZE = stack_size
    self._exp_count = 0

  # adds new experiences into the buffer in the desired format
  # experience in form [st,at,rt,stprime,term]
  def add_exp(self,exp):
    self._buffer.append(exp)
    self._exp_count += 1

  def print_buffer(self):
    print(self._buffer)

  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):
    samples = []
    if self._exp_count == 0 or self._exp_count < self.BATCH_SIZE:
      raise Exception("Experience must be added before it is sampled")
    else:
      for i in range(self.BATCH_SIZE):
        while True:
          index = random.randint(0,self._exp_count)
          if index < self.STACK_SIZE:
            pass
    # Need to find index, look four to left (if able to) and then check that no terminals
    # if terminal redraw index
    # else concatenate into sample
    return samples

      