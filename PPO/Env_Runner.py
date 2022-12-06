import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
# import wandb
# wandb.init(project='PPO SPACE-INVADERS',entity='omg0809')

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
epi_reward=deque([0.0],maxlen=100)

cur_step = 0 
episode = 0          
class Env_Runner:
    def __init__(self, env, agent):
        super().__init__()
        self.agent = agent
        self.env = env
        self.ob = self.env.reset()
        
    def run(self, steps):
        global cur_step
        global episode
        obs = []
        actions = []
        rewards = []
        dones = []
        values = []
        action_prob = []
        
        for step in range(steps):
            self.ob = torch.tensor(self.ob).to(device).to(dtype)
            policy, value = self.agent(self.ob.unsqueeze(0))
            action = self.agent.select_action(policy.detach().cpu().numpy()[0])
            obs.append(self.ob)
            actions.append(action)
            values.append(value.detach())
            action_prob.append(policy[0,action].detach())
            
            self.ob, r, done, info, additional_done = self.env.step(action)
                      
            if done:
                # print("Done")
                self.ob = self.env.reset()
                episode+=1
                if "return" in info:
                    print(" Found the return",episode)
                    epi_reward.append(info['return'])
                    # wandb.log({' return per step :' :info["return"],'Number of steps taken':cur_step+step})
                    if episode%100==0:
                        print('Yes',episode)
                        # wandb.log({'reward per 100 episodes:' : np.mean(epi_reward)})
            rewards.append(r)
            dones.append(done or additional_done)
        cur_step += steps
                                    
        return [obs, actions, rewards, dones, values, action_prob]