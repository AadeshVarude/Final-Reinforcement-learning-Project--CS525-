import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
# import wandb
# wandb.init(project='PPO BREAKOUT Test',entity='omg0809')
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
epi_reward=deque([0.0],maxlen=100)

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()

cur_step = 0 
episode = 0          

class test_Runner:
    
    def __init__(self, env, agent, logger_folder):
        super().__init__()
        
        self.env = env
        self.agent = agent
        
        self.logger = Logger(f'{logger_folder}/testing_info')
        self.logger.log("testing_step, return")
        
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
                    print("per episode reward",info['return'])
                    epi_reward.append(info['return'])
                    # wandb.log({' return per step :' :info["return"],'Number of steps taken':cur_step+step})
                    self.logger.log(f'{cur_step+step},{info["return"]}')
                    if episode%100==0:
                        print('Yes',episode)
                        print('Reward per 100 episodes:', np.mean(epi_reward))
                        break
                        # wandb.log({'reward per 100 episodes:' : np.mean(epi_reward)})

            
            rewards.append(r)
            dones.append(done or additional_done)
            
        cur_step += steps
                                    
        return [obs, actions, rewards, dones, values, action_prob]