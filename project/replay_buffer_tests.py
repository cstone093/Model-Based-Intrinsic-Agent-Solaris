# import os

# from project.datastructures.Replay_Buffer import Replay_Buffer

# current_dir = os.getcwd()
# parent_dir = os.pardir
# rb_dir = os.path.join(parent_dir,"Data_Structures")

from data_structures.replay_buffer import Replay_Buffer

''' Test for replay buffer'''
exp1 = ["s1","r",10,"s2"]
exp2 = ["s2","r",9,"s3"]
exp3 = ["s3","r",8,"s4"]
rb=Replay_Buffer(5,2)
# empty buffer
# print("sample with len 0:",rb.sample()) # works
# only one item in buffer
rb.add_exp(exp1)
print("sample with len 1:",rb.sample())
rb.sample()

# now two in buffer
rb.add_exp(exp2)
print("sample with len 2:",rb.sample())

rb.add_exp(exp3)
rb.add_exp(exp3)
rb.add_exp(exp3)
print("sample with full buffer:",rb.sample())
rb.print_buffer()

rb.add_exp(exp1)
print("overfilled buffer:",rb.sample())
rb.print_buffer()