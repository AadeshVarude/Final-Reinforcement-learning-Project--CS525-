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
import Env_Runner as runner
from Dataset import Batch_Data
# import wandb
# wandb.init(project='PPO SPACE-INVADERS',entity='omg0809')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def advantage_and_value_targets(rewards, values, dones, gamma, lam):
    
    adv_val = []
    old_adv_t = torch.tensor(0.0).to(device)   
    value_targets = []
    old_value_target = values[-1]
    
    for t in reversed(range(len(rewards)-1)): 
        if dones[t]:
            old_adv_t = torch.tensor(0.0).to(device)        
       # ADV
        delta_t = rewards[t] + (gamma*(values[t+1])*int(not dones[t+1])) - values[t]        
        A_t = delta_t + gamma*lam*old_adv_t
        adv_val.append(A_t[0])        
        old_adv_t = delta_t + gamma*lam*old_adv_t        
        # VALUE TARGET
        value_target = rewards[t] + gamma*old_value_target*int(not dones[t+1])
        value_targets.append(value_target[0])        
        old_value_target = value_target
    adv_val.reverse()
    value_targets.reverse()    
    return adv_val, value_targets


def train(args):  
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
    in_channels = num_stacked_frames
    num_actions = gym.make(env_name).env.action_space.n
    agent = PPO_Agent(in_channels, num_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=start_lr)
    env_runners = []

    for actor in range(actors):
        # print('In for actor loop')
        raw_env = gym.make(env_name)
        env = Atari_Games(raw_env, env_name, num_stacked_frames, use_add_done=args.lives)        
        env_runners.append(runner.Env_Runner(env, agent))
        # print(env_runners)        
    num_model_updates = 0
    start_time = time.time()

    while runner.cur_step < total_steps:        
        alpha = 1 - (runner.cur_step / total_steps)
        current_lr = start_lr * alpha
        current_eps = start_eps * alpha
        for g in optimizer.param_groups:
            g['lr'] = current_lr        
        batch_obs, batch_actions, batch_adv, batch_v_t, batch_old_action_prob = None, None, None, None, None

        for env_runner in env_runners:
            # print("In the env runner loop")
            obs, actions, rewards, dones, values, old_action_prob = env_runner.run(T)
            # print("ran 129 steps")
            adv, v_t = advantage_and_value_targets(rewards, values, dones, gamma, lam)    
            
            batch_obs = torch.stack(obs[:-1]) if batch_obs == None else torch.cat([batch_obs,torch.stack(obs[:-1])])
            batch_actions = np.stack(actions[:-1]) if batch_actions is None else np.concatenate([batch_actions,np.stack(actions[:-1])])
            batch_adv = torch.stack(adv) if batch_adv == None else torch.cat([batch_adv,torch.stack(adv)])
            batch_v_t = torch.stack(v_t) if batch_v_t == None else torch.cat([batch_v_t,torch.stack(v_t)]) 
            batch_old_action_prob = torch.stack(old_action_prob[:-1]) if batch_old_action_prob == None else torch.cat([batch_old_action_prob,torch.stack(old_action_prob[:-1])])
    
        
        dataset = Batch_Data(batch_obs,batch_actions,batch_adv,batch_v_t,batch_old_action_prob)
        dataloader = DataLoader(dataset, batch_size=minibatch_size, num_workers=0, shuffle=True)        
        # print('om')
        
        # wandb.log({' Current_eps :' : current_eps,' Current_lr :' : current_lr })

        for epoch in range(epochs):
            #print('epoch',epoch)
            for i, batch in enumerate(dataloader):
                # print(batch)
                # print(batch[0].size())                
                optimizer.zero_grad()
                if i >= 8:
                    break               
                
                obs, actions, adv, v_target, old_action_prob = batch 
                adv = adv.squeeze(1)
                # normalize adv values
                adv = ( adv - torch.mean(adv) ) / ( torch.std(adv) + 1e-8)               
                # get policy actions probs for prob ratio & value prediction
                policy, v = agent(obs)
                # get the correct policy actions
                pi = policy[range(minibatch_size),actions.long()]  
                # probaility ratio r_t(theta)
                probability_ratio = pi / (old_action_prob + 1e-8)                
                # compute CPI
                CPI = probability_ratio * adv
                # compute clip*A_t
                clip = torch.clamp(probability_ratio,1-current_eps,1+current_eps) * adv                     
                # policy loss | take minimum
                L_CLIP = torch.mean(torch.min(CPI, clip))               
                # value loss | mse
                L_VF = torch.mean(torch.pow(v - v_target,2))                
                # policy entropy loss 
                S = torch.mean( - torch.sum(policy * torch.log(policy + 1e-8),dim=1))
                loss = - L_CLIP + c1 * L_VF - c2 * S
                # wandb.log({' loss:' :loss,'adv :':adv})
                loss.backward()
                optimizer.step()
        
            
        num_model_updates += 1
        # print(agent.state_dict())
        # print(agent)    
        # torch.save(agent.state_dict(),'model.pth')
        # torch.save(agent,'full_agent.pth')
        # print time
        if runner.cur_step%50000 < T*actors:
            end_time = time.time()
            print(f'*** total steps: {runner.cur_step} | time(50K): {end_time - start_time} ***')
            start_time = time.time()

        if runner.cur_step%save_model_steps < T*actors:
            torch.save(agent.state_dict(),'modelspace.pth')
            # torch.save(agent,'full_agentspace.pth')

    env.close()
    
if __name__ == "__main__":
    
    args = argparse.ArgumentParser()    
    # set hyperparameter for training chaneg the environment name as per required     
    args.add_argument('-lr', type=float, default=2.5e-4)
    args.add_argument('-env', default='BreakoutNoFrameskip-v4')
    args.add_argument('-lives', type=bool, default=True)
    args.add_argument('-stacked_frames', type=int, default=4)
    args.add_argument('-gamma', type=float, default=0.99)
    args.add_argument('-lam', type=float, default=0.99)
    args.add_argument('-eps', type=float, default=0.1)
    args.add_argument('-c1', type=float, default=1.0)
    args.add_argument('-c2', type=float, default=0.01)
    args.add_argument('-minibatch_size', type=int, default=32)
    args.add_argument('-actors', type=int, default=8)
    args.add_argument('-T', type=int, default=250)
    args.add_argument('-epochs', type=int, default=3)
    args.add_argument('-total_steps', type=int, default=2000000)
    args.add_argument('-save_model_steps', type=int, default=10000)
    args.add_argument('-report', type=int, default=50000)
    
    train(args.parse_args())
