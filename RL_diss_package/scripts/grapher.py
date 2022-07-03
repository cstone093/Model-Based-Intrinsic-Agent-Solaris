import csv
import os
import matplotlib.pyplot as plt
import time

# Get this to take csv as argument and print - get it to work while code is running

def read_rewards(csv_dir):
    f = open(csv_dir)
    results = csv.reader(f, delimiter=",")
    episodes = []
    rewards = []
    frame_count = []

    for row in results:
        episodes.append(row[0])
        rewards.append(row[1])
        frame_count.append(row[2])

    return episodes, rewards, frame_count


def plot_results(csv_dir, log_dir):
    eps, rs, fs = read_rewards(csv_dir)
    # Plot results
    plt.plot(rs)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(log_dir + "/rewards_" + str(time.time())[:10] + ".PNG")

    plt.cla()

    plt.plot(fs)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.savefig(log_dir + "/frames_" + str(time.time())[:10] + ".PNG")


# get pwd for gif store
base_dir = os.getcwd()
log_dir = os.path.join(base_dir, "LOGS")
csv_dir = os.path.join(log_dir, "results.csv")

plot_results(csv_dir, log_dir)