from project.policies.q_value_function import Q_Value_Function
from project.data_structures.replay_buffer import Replay_Buffer
from project.hyperparameters.dqn_hyp import solaris_hyp
from project.environments.solaris import Solaris

exp1 = ["s1",1,"r","s2",False]
exp2 = ["s2",1,"r","s3",False]
exp3 = ["s3",3,"r","s4",False]

env = Solaris()
        
stacked_state, top_state = env.reset()
a,s = env.get_actions_and_obs_shape()

_, reward, term, _, new_top_state, _ = env.step(1)
exp1 = [top_state,1,reward,new_top_state,term]

top_state = new_top_state
_, reward, term, _, new_top_state, _ = env.step(2)
exp2 = [top_state,1,reward,new_top_state,term]

top_state = new_top_state
_, reward, term, _, new_top_state, _ = env.step(3)
exp3 = [top_state,1,reward,new_top_state,term]

q = Q_Value_Function(solaris_hyp,a,s)

rb=Replay_Buffer(5,2,1)
rb.add_exp(exp1)
print("ADDED EXP")
rb.add_exp(exp2)
rb.add_exp(exp3)

sample = rb.sample()
print(sample)