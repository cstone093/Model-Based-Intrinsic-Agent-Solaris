from project.agents.intrinsic_dqn_agent import Intrinsic_DQN
from project.hyperparameters.pong_run import pong
import os
import sys

if len(sys.argv) == 1:
    save_dir = None
else:
    base_dir = os.getcwd()
    save_dir = os.path.join(base_dir, sys.argv[1])

agent = Intrinsic_DQN(hyp=pong,from_file=save_dir,alpha=0.5,beta=0.5)
agent.train()