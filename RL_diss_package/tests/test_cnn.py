from project.data_structures.replay_buffer import Replay_Buffer
from project.policies.q_value_function import Q_Value_Function
from project.hyperparameters.dqn_hyp import solaris_hyp
from project.environments.environment import Environment
import numpy as np

class TestCNN:

    def test_created(self):
        env = Environment(hyp=solaris_hyp)
        a,s = env.get_actions_and_obs_shape()
        q = Q_Value_Function(solaris_hyp,a,s) 
    
    def test_act(self):
        env = Environment(hyp=solaris_hyp)
        
        state = env.reset()
        a,s = env.get_actions_and_obs_shape()

        q = Q_Value_Function(solaris_hyp,a,s)
        action = q.choose_action(state)
        assert action >= 0 and action < a

    def test_learn(self):
        env = Environment(hyp=solaris_hyp)
        
        stacked_state, top_state = env.reset()
        a_shape,s = env.get_actions_and_obs_shape()
        # curr_stacked_state, reward, terminal, life_lost, curr_top_state, unprocessed_new_state
        a = 1
        _ , reward, term, _ ,  new_top_state = env.step(a)
        exp1 = [top_state,a,reward,new_top_state,term]

        q = Q_Value_Function(solaris_hyp,a,s)
        rb=Replay_Buffer(20,3,4)

        for _ in range(12):
            rb.add_exp(exp1)

        states,actions,rewards,new_states,terminals = rb.sample()

        q.learn(states,actions,rewards,new_states,terminals)

    def test_update_target(self):
        env = Environment(hyp=solaris_hyp)
        
        stacked_state, top_state = env.reset()
        a,s = env.get_actions_and_obs_shape()

        q = Q_Value_Function(solaris_hyp,a,s)
        q.update_target_model()

    def test_run(self):
        env = Environment(hyp=solaris_hyp)
        
        stacked_state, top_state = env.reset()
        a,s = env.get_actions_and_obs_shape()
        
        q = Q_Value_Function(solaris_hyp,a,s)
        rb=Replay_Buffer(500,20,4)

        for _ in range(1000):
            action = q.choose_action(stacked_state)
            _ , reward, term, _ ,  new_top_state = env.step(action)
            exp = top_state,action,reward,new_top_state,term

            rb.add_exp(exp)
            top_state = new_top_state

        states,actions,rewards,new_states,terminals = rb.sample()

        q.learn(states,actions,rewards,new_states,terminals)
