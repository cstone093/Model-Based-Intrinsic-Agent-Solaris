from project.agents.dqn_agent import DQN
from project.hyperparameters.dqn_hyp import solaris_hyp

agent = DQN(hyp=solaris_hyp)
agent.train()