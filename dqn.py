import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import deque
import random
import itertools
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
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


class DQN:
    def __init__(self, state_space, action_space):
        self.replay_memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.online_net = Network()
        self.target_net = Network()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.AdamW(self.online_net.parameters(), lr=1e-3)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.epsilon_start = 1
        self.epsilon_end = 0.001
        self.epsilon_decayrate = 0.003

        self.returns = []
    
    def fill_replay_memory(self, env):
        state, _ = env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            s1, reward, done, _, _ = env.step(action)
            experience = (state, action, reward, done, s1)
            self.replay_memory.append(experience)
            state = s1    
            if done:
                env.reset()
    
    def train(self, env):
        self.fill_replay_memory(env)
        for t in range(1000):
            state, _ = env.reset()
            
            for step in itertools.count():
                self.steps += 1
                epsilon = self.decay(self.epsilon_start, self.epsilon_end, self.epsilon_decayrate, self.steps)
                
                if random.random() <= epsilon:
                    action = env.action_space.sample()
                    
                else:
                    action = self.online_net.act(state)
                    
                s1, reward, done, _, _ = env.step(action)
                experience = (state, action, reward, done, s1)
                self.replay_memory.append(experience)
                state = s1
                
                experiences = random.sample(self.replay_memory, self.batch_size)
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
                
                q_values = self.online_net(states_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
                
                target_q_output = self.target_net(new_states_t)
                target_q_values = target_q_output.max(dim=1, keepdim=True)[0]
                optimal_q_values = rewards_t + self.gamma * (1-dones_t) * target_q_values
                
                loss = nn.functional.smooth_l1_loss(action_q_values, optimal_q_values)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.online_net.parameters(), 5)
                self.optimizer.step()
                #self.scheduler.step()

                
                if self.steps % 1000 == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                    
                if done:
                    print(f"Episode {t} finished in {step} steps")
                    self.returns.append(step)
                    break
        
        self.save("dqn.pth")
    
    def render_and_run(self, env):
        state, _ = env.reset()
        while True:
            action = self.online_net.act(state)
            state_p, _, done, _, _ = env.step(action)
            state = state_p
            env.render()
            if done:
                state, _ = env.reset()
    
    def decay(self, start, end, decayrate, current_step):
        return end + (start - end) * np.exp(-decayrate * current_step)
    
    def save(self, path):
        torch.save(self.online_net.state_dict(), path)
        
    def load(self, path):
        self.online_net.load_state_dict(torch.load(path))

