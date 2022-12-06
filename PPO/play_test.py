import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PPO_Agent import PPO_Agent
from Atari_Games import Atari_Games
import test as runner
from Dataset import Batch_Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_test(args):
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    in_channels = num_stacked_frames
    num_actions = gym.make(env_name).env.action_space.n

    agent = PPO_Agent(in_channels, num_actions).to(device)
    checkpoint=torch.load('ppo_breakout_model.pth')
    print("Loaded the model file in the agent")
    agent.load_state_dict(checkpoint)

    raw_env = gym.make(env_name)
    env = Atari_Games(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)
    tr=runner.test_Runner(env, agent)
    tr.run(1000000)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    args.add_argument('-env', default='BreakoutNoFrameskip-v4')
    args.add_argument('-lives', type=bool, default=True)
    args.add_argument('-stacked_frames', type=int, default=4)
    play_test(args.parse_args())
