from collections import deque
import random

class Replay_Buffer():
  def __init__(self,buffer_size,batch_size):
    assert batch_size < buffer_size, "Batch size must be smaller than buffer size"
    assert batch_size > 0, "Batch size must be greater than zero"
    self._buffer = deque(maxlen=buffer_size)
    self.BATCH_SIZE = batch_size
    self._exp_count = 0

  # adds new experiences into the buffer in the desired format
  # exp in form [st,at,rt,stprime]
  def add_exp(self,exp):
    self._buffer.append(exp)
    self._exp_count += 1

  def print_buffer(self):
    print(self._buffer)

  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):
    samples = self._sample_from_buffer()
    processed = []
    for st,at,rt,stprime in samples:
      processed.append({"st":st,"at":at,"rt":rt,"stprime":stprime})
    return processed

  def _sample_from_buffer(self):
    if self._exp_count == 0:
      raise Exception("Experience must be added before it is sampled")
    if self._exp_count < self.BATCH_SIZE:
      return random.sample(self._buffer,self._exp_count)
    else:
      return random.sample(self._buffer,self.BATCH_SIZE)

      