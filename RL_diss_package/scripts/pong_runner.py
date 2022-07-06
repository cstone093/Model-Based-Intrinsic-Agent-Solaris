from project.agents.dqn_agent import DQN
from project.hyperparameters.test_hyp import test_hyp

agent = DQN(hyp=test_hyp)
agent.train()