#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import math
import random
import numpy as np


from collections import namedtuple
from itertools import count
from PIL import Image
import itertools
from collections import deque



env = gym.make('CartPole-v1')
env.reset()


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 2))
        
    def forward(self, t):
        t = self.net(t)
        return t
    
    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))
        
        max_q = torch.argmax(q_values, dim=1)[0]
        action = max_q.item()
        
        return action


def decay(eps_start, eps_end, eps_decayrate, current_step):
    return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decayrate * current_step)


eps_count = 0
batch_size = 32
gamma = 0.99

online_net = Network()
target_net = Network()
target_net.load_state_dict(online_net.state_dict())

epsilon_start = 1
epsilon_end = 0.001
epsilon_decayrate = 0.003

epsiode_durations = []

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)
stop = 0


replayMemory = deque(maxlen=50000)
state = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    s1, reward, done, __ = env.step(action)
    experience = (state, action, reward, done, s1)
    replayMemory.append(experience)
    state = s1
    
    if done:
        env.reset()
        
for t in range(1000):
    state = env.reset()
    
    for step in itertools.count():
        eps_count += 1
        epsilon = decay(epsilon_start, epsilon_end, epsilon_decayrate, eps_count)
        
        if random.random() <= epsilon:
            action = env.action_space.sample()
            
        else:
            action = online_net.act(state)
            
        s1, reward, done, __ = env.step(action)
        experience = (state, action, reward, done, s1)
        replayMemory.append(experience)
        state = s1
        
        experiences = random.sample(replayMemory, batch_size)
        states = np.asarray([e[0] for e in experiences])
        actions = np.asarray([e[1] for e in experiences])
        rewards = np.asarray([e[2] for e in experiences])
        dones = np.asarray([e[3] for e in experiences])
        new_states = np.asarray([e[4] for e in experiences])
        
        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_states_t = torch.as_tensor(new_states, dtype=torch.float32)
        
        q_values = online_net(states_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
        
        target_q_output = target_net(new_states_t)
        target_q_values = target_q_output.max(dim=1, keepdim=True)[0]
        optimal_q_values = rewards_t + gamma * (1-dones_t) * target_q_values
        
        loss = nn.functional.smooth_l1_loss(action_q_values, optimal_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if eps_count % 1000 == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        if done:
            print(step)
            break
            
        if step>= 170:
            stop += 1
        if stop >=3:
            while True:
                action = online_net.act(state)
                state, _, done, _ = env.step(action)
                env.render()
                if done:
                    state = env.reset()
