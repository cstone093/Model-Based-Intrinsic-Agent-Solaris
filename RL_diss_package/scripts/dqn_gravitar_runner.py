from project.agents.dqn_agent import DQN
from project.hyperparameters.gravitar_hyp import gravitar_hyp

agent = DQN(hyp=gravitar_hyp)
agent.train()