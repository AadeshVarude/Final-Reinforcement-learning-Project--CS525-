import numpy as np
import cv2
import gym

class Atari_Games(gym.Wrapper):
    def __init__(self, env, env_name, k, dsize=(84,84), use_add_done=False):
        super(Atari_Games, self).__init__(env)
        self.dsize = dsize
        self.k = k
        # self.use_add_done = use_add_done
        if "Breakout" in env_name:
            self.frame_cutout_h = (31,-16)
            self.frame_cutout_w = (7,-7)
        elif "SpaceInvaders" in env_name:
            self.frame_cutout_h = (25,-7)
            self.frame_cutout_w = (7,-7)
        
    def reset(self):
        self.Return = 0
        self.last_life_count = 0
        ob = self.env.reset()
        ob = self.preprocess_observation(ob)
        self.frame_stack = np.stack([ob for i in range(self.k)])
        
        return self.frame_stack
    
    
    def step(self, action): 
        reward = 0
        done = False
        additional_done = False
        frames = []
        for i in range(self.k):
            ob, r, d, info = self.env.step(action)
            ob = self.preprocess_observation(ob)
            frames.append(ob)
            reward += r
            
            if d: # env done
                done = True
                break
                       
        # build the observation
        self.step_frame_stack(frames)
        
        # add info, get return of the completed episode
        self.Return += reward
        if done:
            info["return"] = self.Return
            
        # clip reward
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = 0
        else:
            reward = -1
            
        return self.frame_stack, reward, done, info, additional_done
    
    def step_frame_stack(self, frames):
        
        num_frames = len(frames)
        
        if num_frames == self.k:
            self.frame_stack = np.stack(frames)
        elif num_frames > self.k:
            self.frame_stack = np.array(frames[-k::])
        else: 
            
            # shift the existing frames in the framestack to the front=0 (0->k, index is time)
            self.frame_stack[0: self.k - num_frames] = self.frame_stack[num_frames::]
            # insert the new frames into the stack
            self.frame_stack[self.k - num_frames::] = np.array(frames)  
            
    def preprocess_observation(self, ob):
    
        ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
                           self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        ob = cv2.resize(ob, dsize=self.dsize)
    
        return ob