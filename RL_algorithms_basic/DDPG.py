"""
Author: David Valencia
Date: 08/ 04 /2022

Describer:
            Basic implementation from scratch of Deep Deterministic Policy Gradient (DDPG)
            Need:
                Python 3
                Pytorch
                Gym Env --> Pendulum-v1

            Important ideas:
            - Continuous action space only
            - Exploration is done by adding Noise to the action
            - Action Space  --> representing the torque, Number of action: one action
            - Observation Space  --> representing the x-y coordinates of the pendulum's end and its angular velocity
            - Reward --> *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
            - The episode terminates at 200 time steps.
"""

import sys
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.h_linear_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # Concatenates the seq tensors in the given dimension
        x = F.relu(self.h_linear_1(x))
        x = F.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)  # no activation function here
        return x


class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(input_size, hidden_size)
        self.h_linear_2 = nn.Linear(hidden_size, hidden_size)
        self.h_linear_3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))
        x = torch.tanh(self.h_linear_3(x))
        return x


class OUNoise(object):
    # Ornstein-Uhlenbeck Process to add noise to the action output
    # just copied and pasted from the source code
    # to encourage action exploration

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Memory:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class DDPGagent:

    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                       gamma=0.99, tau=1e-2, max_memory_size=50000):

        self.num_states = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]      # 1

        self.gamma = gamma
        self.tau = tau
        self.t_step = 0  # counter for activating learning every few steps

        # Initialization of the networks
        self.actor  = Actor(self.num_states, hidden_size, self.num_actions)  # main network Actor
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)  # main network Critic

        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Initialization memory
        self.memory = Memory(max_memory_size)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state_chan = torch.from_numpy(state).float().unsqueeze(0)  # a tensor with shape [1,3]
        state_ready = Variable(state_chan)  # need Variable to perform backpropagation

        action = self.actor.forward(state_ready)
        action = action.detach()
        action = action.numpy()
        return action[0]

    def step_training(self, batch_size):
        LEARN_EVERY_STEP = 100
        self.t_step = self.t_step + 1

        if self.t_step % LEARN_EVERY_STEP == 0:
            if self.memory.__len__() > batch_size:
                self.learn_step(batch_size)


    def learn_step(self, batch_size):

        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Calculate the Critic loss
        Q_vals = self.critic.forward(states, actions)

        next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target
        next_Q_vales = self.critic_target.forward(next_states, next_actions.detach())  # Note this is from critic-target

        Q_target = rewards + self.gamma * next_Q_vales

        loss = nn.MSELoss()
        critic_loss = loss(Q_vals, Q_target)

        # Calculate the Actor loss
        actor_loss = - self.critic.forward(states, self.actor.forward(states)).mean()

        # ----- update networks ----- #
        # Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
        self.actor_optimizer.step()

        # Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        # there is a rule for this check the literature "soft updates"

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def main():

    EPISODE = 50000
    batch_size = 64
    # -------------------------------
    env = gym.make('Pendulum-v1')
    agent = DDPGagent(env)
    noise = OUNoise(env.action_space)

    # -------------------------------
    rewards = []
    avg_rewards = []
    # -------------------------------
    for episode in range(1, EPISODE):
        done = False
        state = env.reset()
        noise.reset()
        episode_reward = 0

        for step in range(200):
            env.render()
            action = agent.get_action(state)
            action = noise.get_action(action, step)  # exploration is done via adding noise to the action itself
            new_state, reward, done, _ = env.step(action)

            agent.memory.push(state, action, reward, new_state, done)
            agent.step_training(batch_size)

            state = new_state
            episode_reward += reward

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode,
                                                                                          np.round(episode_reward,
                                                                                                   decimals=2),
                                                                                          np.mean(rewards[-10:])))
                break

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))

    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


if __name__ == '__main__':
    main()
