import numpy as np 
import gym 
import torch 
import random
from argparse import ArgumentParser 
import os 
import pandas as pd 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
from scipy.ndimage.filters import gaussian_filter1d


def arguments(): 

    parser = ArgumentParser()
    parser.add_argument('--env', default = 'BipedalWalker-v2')

    return parser.parse_args()


def save(agent, rewards, args): 

    path = './runs/{}/'.format(args.env)
    try: 
        os.makedirs(path)
    except: 
        pass 

    torch.save(agent.q.state_dict(), os.path.join(path, 'model_state_dict'))

    plt.cla()
    plt.plot(rewards, c = 'r', alpha = 0.3)
    plt.plot(gaussian_filter1d(rewards, sigma = 5), c = 'r', label = 'Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.title('Branching DDQN: {}'.format(args.env))
    plt.savefig(os.path.join(path, 'reward.png'))

    pd.DataFrame(rewards, columns = ['Reward']).to_csv(os.path.join(path, 'rewards.csv'), index = False)





class AgentConfig:

    def __init__(self, 
                 epsilon_start = 1.,
                 epsilon_final = 0.01,
                 epsilon_decay = 8000,
                 gamma = 0.99, 
                 lr = 1e-4, 
                 target_net_update_freq = 1000, 
                 memory_size = 100000, 
                 batch_size = 128, 
                 learning_starts = 5000,
                 max_frames = 10000000): 

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma =gamma
        self.lr =lr

        self.target_net_update_freq =target_net_update_freq
        self.memory_size =memory_size
        self.batch_size =batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class TensorEnv(gym.Wrapper): 

    def __init__(self, env_name): 

        super().__init__(gym.make(env_name))

    def process(self, x): 

        return torch.tensor(x).reshape(1,-1).float()

    def reset(self): 

        return self.process(super().reset())

    def step(self, a): 

        ns, r, done, infos = super().step(a)
        return self.process(ns), r, done, infos 


class BranchingTensorEnv(TensorEnv): 

    def __init__(self, env_name, n): 

        super().__init__(env_name)
        self.n = n 
        self.discretized = np.linspace(-1.,1., self.n)


    def step(self, a):

        action = np.array([self.discretized[aa] for aa in a])

        return super().step(action)
