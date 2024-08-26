import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from envs.mavic_env import MavicEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu

environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
action_dim = 2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/actor.pth" % (directory))
        )

def main():
    # Create the network
    network = TD3(state_dim, action_dim)
    try:
        network.load("./pytorch_models")
    except:
        raise ValueError("Could not load the stored model parameters")

    # Initialize the environment
    env = MavicEnv()

    obs = env.reset()
    for z in range(100000):
        obs_copy = obs.copy()

        ### Debug: fill lidar data with 3
        # obs_copy[:-4] = 3

        ### Debug: print observation
        # print("obs:", obs_copy[-4:])
        # print(obs_copy[:-4])
        # print()

        action = network.get_action(obs_copy)
        a_in = [(action[0] + 1) / 2, -action[1]]
        # a_in = [0, 0.5]
        obs, reward, done, info = env.step(a_in)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()