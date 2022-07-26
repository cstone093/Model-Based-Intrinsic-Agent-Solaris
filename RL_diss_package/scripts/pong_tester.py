from project.agents.dqn_agent import DQN
from project.hyperparameters.test_hyp import test_hyp
import os
import sys

if len(sys.argv) == 1:
    save_dir = None
else:
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, sys.argv[1])

agent = DQN(hyp=test_hyp,from_file=save_dir)
agent.train()