from collections import namedtuple
import os
import numpy as np
import csv
import random
import pandas as pd 
from numpy import asarray, ndarray
from numpy import savetxt

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

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

    self._transitions = np.empty(self.BUFFER_SIZE,dtype=Experience)

    self._exp_count = 0
    self._new_index = 0

  # adds new experiences into the buffer from the format
  # <st,at,rt,stprime,term>
  def add_exp(self,s,a,r,sp,d):
    exp = Experience(s,a,r,d,sp)
    self._transitions[self._new_index] = exp

    self._exp_count = np.max([self._exp_count, self._new_index + 1])
    self._new_index = (self._new_index + 1) % self.BUFFER_SIZE

  # samples from the buffer and processes the sample so it is in the desired form for the agent
  def sample(self):
    indices = np.random.choice(self._exp_count, self.BATCH_SIZE, replace=False)
    states, actions, rewards, dones, next_states = zip(*[self._transitions[idx] for idx in indices])
    return np.array(states,dtype=np.float32), np.array(actions,dtype=np.uint8), np.array(rewards, dtype=np.float32), \
           np.array(next_states), np.array(dones, dtype=np.bool8)
    
  def save_state(self,directory):
    # Save content of replay buffer
    rb_state_log_path = os.path.join(directory, "replay_buffer_state.csv")
    rb_new_state_log_path = os.path.join(directory, "replay_buffer_new_state.csv")
    rb_other_log_path = os.path.join(directory, "replay_buffer_other.csv")

    states = []
    new_states = []
    for exp in self._transitions:
      if exp is not None:
          states.append(ndarray.flatten(exp[0]))
          new_states.append(ndarray.flatten(exp[4]))
      else:
        continue
    states = ndarray.flatten(np.array(states))

    with open(rb_state_log_path, 'a',newline="\n") as a:
      a_out = csv.writer(a, delimiter =',')
      a_out.writerow(states)

    with open(rb_new_state_log_path, 'a',newline="\n") as c:
      b_out = csv.writer(c, delimiter =',')
      b_out.writerow(new_states)

    # df_a = pd.DataFrame(states,columns=["state"])
    # print(df_a)
    # df_a.to_csv(rb_state_log_path)

    # df_b = pd.DataFrame(new_states,columns=["new_state"])
    # df_b.to_csv(rb_new_state_log_path)

    with open(rb_other_log_path, 'a',newline="\n") as b:
      c_out = csv.writer(b, delimiter =',')
      for i, exp in enumerate(self._transitions):
        if exp is not None:
          c_out.writerow([exp[1],exp[2],exp[3]])
        else:
          continue
        
    # Save parameters
    param_log_path = os.path.join(directory, "params.csv")
    with open(param_log_path, 'a',newline="\n") as out:
      csv_out = csv.writer(out, delimiter =',')
      csv_out.writerow([self._exp_count,self._new_index])          
    print(f"Successfully saved replay buffer")

  def load_state(self,directory):
    rb_state_log_path = os.path.join(directory, "replay_buffer_state.csv")
    rb_new_state_log_path = os.path.join(directory, "replay_buffer_new_state.csv")
    rb_other_log_path = os.path.join(directory, "replay_buffer_other.csv")
     
    a = pd.read_csv(rb_state_log_path, sep=',',lineterminator="\n")
    b = pd.read_csv(rb_new_state_log_path, sep=',',lineterminator="\n")

    print(a)

    with open(rb_other_log_path) as f:
      reader = csv.reader(f, delimiter =',')
      i=0
      for row in reader:
        if i == 0:
          self._exp_count = int(row[0])
          self._new_index = int(row[1])
        else:
          continue
        i+=1

    rb_log_path = os.path.join(directory, "replay_buffer.csv")
     

    param_log_path = os.path.join(directory, "params.csv")
    with open(param_log_path) as f:
      reader = csv.reader(f, delimiter =',')
      i=0
      for row in reader:
        if i == 0:
          self._exp_count = int(row[0])
          self._new_index = int(row[1])
        else:
          continue
        i+=1
    print(f"Successfully loaded replay buffer from {param_log_path} and {rb_log_path}")

