from tqdm import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 

import numpy as np 
import gym 
import random 

from model import DuelingNetwork, BranchingQNetwork
from utils import TensorEnv, ExperienceReplayMemory, AgentConfig, BranchingTensorEnv
import utils

class BranchingDQN(nn.Module): 

    def __init__(self, obs, ac, n, config): 

        super().__init__()

        self.q = BranchingQNetwork(obs, ac,n )
        self.target = BranchingQNetwork(obs, ac,n )

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim = 1)
        return action.numpy()

    def update_policy(self, adam, memory, params): 

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0],-1,1)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1,1)

        qvals = self.q(states)


        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():


            argmax = torch.argmax(self.q(next_states), dim = 2)

            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim = True)

        expected_q_vals = rewards + max_next_q_vals*0.99*masks
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)

        # input(loss)

        # print('\n'*5)
        
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0: 
            self.update_counter = 0 
            self.target.load_state_dict(self.q.state_dict())


args = utils.arguments()

bins = 6
env = BranchingTensorEnv(args.env, bins)
        
config = AgentConfig()
memory = ExperienceReplayMemory(config.memory_size)
agent = BranchingDQN(env.observation_space.shape[0], env.action_space.shape[0], bins, config)
adam = optim.Adam(agent.q.parameters(), lr = config.lr) 


s = env.reset()
ep_reward = 0. 
recap = []

p_bar = tqdm(total = config.max_frames)
for frame in range(config.max_frames): 

    epsilon = config.epsilon_by_frame(frame)

    if np.random.random() > epsilon: 
        action = agent.get_action(s)
    else: 
        action = np.random.randint(0, bins, size = env.action_space.shape[0])

    ns, r, done, infos = env.step(action)
    ep_reward += r 

    if done:
        ns = env.reset()
        recap.append(ep_reward)
        p_bar.set_description('Rew: {:.3f}'.format(ep_reward))
        ep_reward = 0.  

    memory.push((s.reshape(-1).numpy().tolist(), action, r, ns.reshape(-1).numpy().tolist(), 0. if done else 1.))
    s = ns  

    p_bar.update(1)

    if frame > config.learning_starts:
        agent.update_policy(adam, memory, config)

    if frame % 1000 == 0: 
        utils.save(agent, recap, args)



p_bar.close()
