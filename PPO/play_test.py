import numpy as np
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PPO_Agent import PPO_Agent
from Atari_Games import Atari_Games
import test as runner
from Dataset import Batch_Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_test(args):
    folder_name = time.asctime(time.gmtime()).replace(" ","_").replace(":","_")
    os.mkdir(folder_name)
    
    # save the hyperparameters in a file
    f = open(f'{folder_name}/args.txt','w')
    for i in args.__dict__:
        f.write(f'{i},{args.__dict__[i]}\n')
    f.close()
        # arguments
    env_name = args.env
    num_stacked_frames = args.stacked_frames
    start_lr = args.lr 
    gamma = args.gamma
    lam = args.lam
    minibatch_size = args.minibatch_size
    T = args.T
    c1 = args.c1
    c2 = args.c2
    actors = args.actors
    start_eps = args.eps
    epochs = args.epochs
    total_steps = args.total_steps
    save_model_steps = args.save_model_steps
     # in/output    
    in_channels = num_stacked_frames
    num_actions = gym.make(env_name).env.action_space.n

    agent = PPO_Agent(in_channels, num_actions).to(device)
    checkpoint=torch.load('model.pth')
    print("Loaded the model file in the agent")
    agent.load_state_dict(checkpoint)

    raw_env = gym.make(env_name)
    env = Atari_Games(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)
    
    tr=runner.test_Runner(env, agent, folder_name)
    obs, actions, rewards, dones, values, old_action_prob = tr.run(1000)

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    
    # set hyperparameter
    
    args.add_argument('-lr', type=float, default=2.5e-4)
    args.add_argument('-env', default='BreakoutNoFrameskip-v4')
    args.add_argument('-lives', type=bool, default=True)
    args.add_argument('-stacked_frames', type=int, default=4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-lam', type=float, default=0.95)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-actors', type=int, default=8)
    args.add_argument('-T', type=int, default=129)
    args.add_argument('-epochs', type=int, default=4)
    args.add_argument('-total_steps', type=int, default=10000000)
    args.add_argument('-save_model_steps', type=int, default=1000000)
    args.add_argument('-report', type=int, default=50000)
    
    play_test(args.parse_args())
