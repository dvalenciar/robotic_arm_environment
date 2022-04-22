"""
Author: David Valencia
Date: 12/ 04 /2022
Completed :

Describer:
            Distributed Distributional Deterministic Policy Gradients (D4PG)

            A D4PG algorithm attempts to improve the accuracy of a
            DDPG algorithm by incorporating a distributional approach and N-step return

            Need:
                Python 3
                Pytorch
                Gym Env --> Pendulum-v1

            Important points:
            - Continuous action space only
            - Action Space  --> representing the torque, Number of action: one action
            - Observation Space  --> representing the x-y coordinates of the pendulum's end and its angular velocity
            - Reward --> *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
            - The episode automatically terminates at 200 time steps.
            - D4PG uses simple random noise from normal distribution to encourage action exploration instead of OU noise

            - D4PG can use K agent running in parallet
            - Actor and critic NN are update vary different wrt DDPG, here distribution to distributions updates


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

    def __init__(self, input_size, hidden_size, num_atoms):
        super(Critic, self).__init__()

        self.h_linear_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.h_linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.h_linear_3 = nn.Linear(in_features=hidden_size, out_features=num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)  # No activation function here
        x = F.softmax(x, dim=-1)  # Because critic should output probabilities
        return x


class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(input_size, hidden_size)
        self.h_linear_2 = nn.Linear(hidden_size, hidden_size)
        self.h_linear_3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.tanh(self.h_linear_3(x))
        return x


class NoiseGenerator:

    def __init__(self, action_dims, action_bound_high, noise_scale=0.3):

        self.action_dims = action_dims
        self.action_bounds = action_bound_high
        self.noise_scale = noise_scale

    def noise_gen(self):
        noise = np.random.normal(size=self.action_dims) * self.action_bounds * self.noise_scale
        return noise


class Memory:

    def __init__(self, replay_max_size, rollout_len=5):

        self.replay_max_size = replay_max_size
        self.rollout_len = rollout_len  # this is the N on the paper

        self.replay_buffer = deque(maxlen=replay_max_size)  # replay buffer, batch of memories to sample during training
        self.trajectories  = deque(maxlen=rollout_len)  # esto es para guardar N transitions, creo

        self.special_buffer = deque(maxlen=replay_max_size)

    def buffer_add(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.replay_buffer.append(experience)

    def trajectory_add(self, state, action, reward, next_state, done):
        # incorporates computing of n-steps instead of having a one-step reward
        n_steps_experience = (state, action, np.array([reward]), next_state, done)
        self.trajectories.append(n_steps_experience)

    def push_special(self, sequence_trajectories):
        experience = sequence_trajectories
        self.special_buffer.append(experience)

    def special_sample_experience(self, batch_size):

        n_step_batch = random.sample(self.special_buffer, batch_size)

        n_step_state_batch  = []
        n_step_action_batch = []
        n_step_reward_batch = []
        n_step_next_state_batch = []
        n_step_done_batch = []

        for n_step_trajectory in n_step_batch:
            trajectory_rew = 0
            n = 0

            for single_trajectory in n_step_trajectory:

                state, action, reward, next_state, done = single_trajectory

                if n < len(n_step_trajectory)-1:
                    trajectory_rew += reward  # todo multiplicate this for gama
                else:
                    last_state  = state  # last state_ in n_step_batch
                    last_action = action
                    last_next_state = next_state
                    last_done = done
                n += 1

            n_step_reward_batch.append(trajectory_rew)  #
            n_step_state_batch.append(last_state)  # save the last state in the single trajectory i.e  Xi+N
            n_step_action_batch.append(last_action)  # need to be here cause we only need the last act in the trajectory
            n_step_next_state_batch.append(last_next_state)
            n_step_done_batch.append(last_done)

        return n_step_state_batch, n_step_action_batch, n_step_reward_batch, n_step_next_state_batch, n_step_done_batch

    def sample_experience(self, batch_size):
        state_batch  = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.replay_buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.replay_buffer)


class D4PGAgent:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99,
                 max_memory_size=50000, rollout_len=5, tau=1e-2):

        # -------- Parameters --------------- #
        self.num_states  = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]  # 1

        self.act_max_bound = env.action_space.high  # [2.]
        self.act_min_bound = env.action_space.low  # [-2.]

        # these parameters are used for the distribution
        self.n_atoms = 51  # paper define this parameter as l
        self.v_min = -10
        self.v_max = 10
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin = torch.linspace(self.v_min, self.v_max, self.n_atoms).view(-1, 1)

        self.gamma = gamma
        self.tau = tau

        # ---------- Initialization and build the networks ----------- #
        hidden_size = 256  # todo try different size for each layer
        self.actor  = Actor(self.num_states, hidden_size, self.num_actions)  # main network Actor
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)  # main network Critic

        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # ------------- Initialization memory --------------------- #
        self.memory = Memory(max_memory_size, rollout_len)

        #  ----------- Gaussian Noise Generator -------------- #
        self.noise = NoiseGenerator(self.num_actions, self.act_max_bound)

    def get_action(self, state):

        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
            action = action.detach()
            action = action.numpy()
            noise = np.random.normal(size=action.shape)
            action = np.clip(action + noise, -1, 1)  # todo maybe I could change this to -2, 2 .....?
        self.actor.train()

        return action[0]

    def distribution_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma):

        # todo maybe need to conver to numpy first
        next_distr = next_distr_v
        rewards = rewards_v
        dones_mask = dones_mask_t
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.n_atoms), dtype=np.float32)

        for atom in range(self.n_atoms):

            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * gamma))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():

            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[dones_mask]))
            b_j = (tz_j - self.v_min) / self.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
        return torch.FloatTensor(proj_distr)

    def linear_interpolation(self, next_distribution, rewards, dones):
        """
        A projection Phi to linearly interpolate the target atoms with their neighbors in Z
        """
        gamma = 0.99
        delta_z = self.delta
        batch_size = rewards.size(0)

        m = torch.zeros(batch_size, self.n_atoms)  # projection distribution
        support = torch.arange(self.v_min, self.v_max + self.delta, self.delta)

        for sample_idx in range(batch_size):
            reward = rewards[sample_idx]
            done = dones[sample_idx]

            for atom in range(self.n_atoms):
                Tz_j = reward + (1 - done) * gamma * support[atom]
                Tz_j = torch.clamp(Tz_j, self.v_min, self.v_max)
                b_j = (Tz_j - self.v_min) / delta_z
                l = torch.floor(b_j).long().item()
                u = torch.ceil(b_j).long().item()

                # distribute probability of Tz_j
                m[sample_idx][l] = m[sample_idx][l] + next_distribution[sample_idx][atom] * (u - b_j)
                m[sample_idx][u] = m[sample_idx][u] + next_distribution[sample_idx][atom] * (b_j - l)

        return m

    def update(self, batch_size):

        states, actions, rewards, next_states, done = self.memory.special_sample_experience(batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones   = np.array(done)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones   = torch.FloatTensor(dones)
        next_states = torch.FloatTensor(next_states)

        # calculate the distribution prediction Z(s,a) --> Q_values
        Z_val = self.critic.forward(states, actions)  # this is a categorical distribution, the Z predicted

        # calculate the next Z distribution Z(s',a') --> Q_next_value
        next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target
        next_Z_val   = self.critic_target.forward(next_states, next_actions.detach())  # Note this is from critic-target

        # calculate the project target distribution Y --> Q_target
        projected_distribution = self.linear_interpolation(next_Z_val, rewards, dones)
        Y = projected_distribution

        # ----------------------------------- Calculate the loss ----- #

        # ------- calculate the critic loss
        loss = torch.nn.BCELoss(reduction='none')
        critic_loss = loss(Z_val, Y)
        critic_loss = critic_loss.mean(axis=1)
        critic_loss = critic_loss.mean()

        # ------- calculate the actor loss
        z_atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
        z_atoms = torch.from_numpy(z_atoms).float()

        actor_loss = self.critic.forward(states, self.actor.forward(states))
        actor_loss = actor_loss * z_atoms
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()

        # ------------------------------------- Update networks ----- #

        # Actor step Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic step Update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update the target networks using tao "soft updates"

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def main():

    EPISODES = 2     # ---> T, total number of episodes
    n_steps = 200    # number of steps per Episode
    batch_size = 2   # ---> M
    rollout_len = 5  # ---> N
    # -------------------------------
    transition_vector = deque(maxlen=rollout_len)
    # -------------------------------
    env = gym.make('Pendulum-v1')
    agent = D4PGAgent(env)
    state = env.reset()
    # -------------------------------
    for episode in range(1, 2):
        for step in range(3):
            for n in range(1, rollout_len+1):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                transition = (state, action, np.array([reward]), next_state, done)
                transition_vector.append(transition)
                state = next_state
                # todo check the performance without this
            agent.memory.push_special(transition_vector)

        if len(agent.memory.special_buffer) > batch_size:
            print("oki")
            agent.update(batch_size)


if __name__ == '__main__':
    main()
