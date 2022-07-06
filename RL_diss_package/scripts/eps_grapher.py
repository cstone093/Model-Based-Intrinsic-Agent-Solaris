from project.policies.q_value_function import Q_Value_Function
from project.hyperparameters.dqn_hyp import solaris_hyp
from project.hyperparameters.test_hyp import test_hyp
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

hyp_choice = sys.argv[1]

if hyp_choice == "test": hyp = test_hyp
else: hyp = solaris_hyp

base_dir = os.getcwd()
log_dir = os.path.join(base_dir, "logs")

ls = np.linspace(0,hyp["EPS_FRAMES_FINAL"],10_000)
nn = Q_Value_Function(hyp=test_hyp,a_size=6,s_size=(84,84,5))
ys = []
for l in ls:
    ys.append(nn.get_epsilon(l))

plt.plot(ls,ys)
plt.xlabel("Frame")
plt.ylabel("Epsilon")
plt.axvline(x = hyp["EPS_FRAMES_INIT"], color = 'b', label = 'intermediate decay')
plt.axvline(x = hyp["EPS_FRAMES_INTER"], color = 'b', label = 'final decay')
plt.legend()
plt.title("Rate of Exploration with increase of Training Frames")
plt.savefig(log_dir + "/epsilon_decay" + str(time.time())[:10] + ".PNG")
print("Generated episode v episode length graph")