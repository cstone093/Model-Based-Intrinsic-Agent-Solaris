from project.data_structures.replay_buffer import Replay_Buffer
from project.policies.q_value_function import Q_Value_Function
from project.hyperparameters.DQN_HP import solaris_hyp
from project.environments.solaris import Solaris
import numpy as np

class TestCNN:

    def test_created(self):
        env = Solaris()
        a,s = env.get_actions_and_obs_shape()
        q = Q_Value_Function(solaris_hyp,a,s) 
    
    def test_act(self):
        env = Solaris()
        
        state = env.reset()
        a,s = env.get_actions_and_obs_shape()

        q = Q_Value_Function(solaris_hyp,a,s)
        action = q.choose_action(state)
        assert action > 0 and action < a

    def test_learn(self):
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

        rb=Replay_Buffer(5,2)
        rb.add_exp(exp1)
        rb.add_exp(exp2)
        rb.add_exp(exp3)

        sample = rb.sample()

        q.learn(sample)

        # NEED TO CHECK that exception is raised if not enough experience - and handle this in cnn 

    def test_forward(self):
        pass
