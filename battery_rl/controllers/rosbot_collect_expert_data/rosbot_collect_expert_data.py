import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from envs.rosbot_env import RosbotEnv


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

    def load(self, model_name):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load(f"./pytorch_models/{model_name}.pth")
        )

def main():
    model_name = 'expert'
    X, Y = [], []

    # Create the network
    network = TD3(state_dim, action_dim)
    try:
        network.load(model_name)
    except:
        raise ValueError("Could not load the stored model parameters")

    # Initialize the environment
    env = RosbotEnv()
    obs = env.reset()
    x, y = [], []
    while True:
        obs_copy = obs.copy()
        ### Debug: fill lidar data with 3
        obs_copy[:-4] = 3
        ### Debug: print observation
        # print("obs:", obs_copy[-4:])
        # print(obs_copy[:-4])
        # print()
        x.append(obs)
        action = network.get_action(obs_copy)
        y.append(action)
        a_in = [(action[0] + 1) / 2, action[1]]

        obs, reward, done, info = env.step(a_in)
        if done:
            # print(len(X), len(Y), len(x), len(y))
            if info['target']:
                X += x
                Y += y
            x = []
            y = []
            if len(X) >= 5000:
                break
            # print(len(X), len(Y), len(x), len(y))
            # print('='*30)
            print(len(X))
            obs = env.reset()

    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    np.savez('pytorch_models/data.npz', X=X, Y=Y)

if __name__ == '__main__':
    main()