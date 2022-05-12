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
            - D4PG can use K agent running in parallet (However, here I use only 1 agent)
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

        self.h_linear_1 = nn.Linear(in_features=input_size, out_features=256)
        self.h_linear_2 = nn.Linear(in_features=256, out_features=128)
        self.h_linear_3 = nn.Linear(in_features=128, out_features=num_atoms)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenates the seq tensors in the given dimension
        x = torch.relu(self.h_linear_1(x))
        x = torch.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)  # No activation function here
        x = F.softmax(x, dim=1)  # softmax because critic should output probabilities
        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        self.h_linear_1 = nn.Linear(input_size, 256)
        self.h_linear_2 = nn.Linear(256, 128)
        self.h_linear_3 = nn.Linear(128, output_size)

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
    def __init__(self, replay_max_size):
        self.replay_max_size = replay_max_size
        self.replay_buffer = deque(maxlen=replay_max_size)  # batch of experiences to sample during training

    def replay_buffer_add(self, state, action, reward, next_state, done):
        #experience = (state, action, np.array([reward]), next_state, done)
        experience = (state, action, reward, next_state, done)
        self.replay_buffer.append(experience)

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


class PerMemory(object):
    # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4

    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    def per_add(self, state, action, reward, next_state, done):
        #experience = state, action, np.array([reward]), next_state, done
        experience = state, action, (reward), next_state, done
        self.store(experience)

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience
        # will never have a chance to be selected, so we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    def sample_experience(self, n):
        # Create a minibatch array that will contain the minibatch of experiences
        minibatch = []
        b_idx = np.empty((n,), dtype=np.int32)
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            b_idx[i] = index
            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumTree(object):
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1)
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Look at what index we want to put the experience, fill the leaves from left to right
        tree_index = self.data_pointer + self.capacity - 1
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class D4PGAgent:

    def __init__(self, env, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.99,
                 max_memory_size=50000, tau=1e-3, n_steps=1):

        # -------- Parameters --------------- #
        self.num_states  = env.observation_space.shape[0]  # 3
        self.num_actions = env.action_space.shape[0]  # 1

        self.act_max_bound = env.action_space.high  # [2.]
        self.act_min_bound = env.action_space.low   # [-2.]

        # these parameters are used for the probability distribution
        self.n_atoms = 51
        self.v_min   = -10
        self.v_max   = 10
        self.delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.v_lin = torch.linspace(self.v_min, self.v_max, self.n_atoms).view(-1, 1)

        self.gamma = gamma  # discount factor
        self.tau = tau
        self.n_steps = n_steps
        self.t_step = 0  # counter for activating learning every few steps

        # ---------- Initialization and build the networks ----------- #
        hidden_size = 256  # todo try different size for each hidden layer
        self.actor  = Actor(self.num_states, hidden_size, self.num_actions)  # main Actor network Actor
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)  # main Critic network

        self.actor_target  = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.n_atoms)

        # Initialization of the target networks as copies of the original networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # optimizers
        # todo check with different lr values
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        #  ----------- Gaussian Noise Generator -------------- #
        self.noise = NoiseGenerator(self.num_actions, self.act_max_bound)

        # ------------- Initialization memory --------------------- #
        self.memory = Memory(max_memory_size)

        per_max_memory_size = 10000
        self.memory_per = PerMemory(per_max_memory_size)

    def get_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # numpy to a tensor with shape [1,3]

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
            action = action.detach()
            action = action.numpy()
            noise = np.random.normal(size=action.shape)
            action = np.clip(action + noise, -1, 1)  # todo maybe I could change this to -2, 2?
        self.actor.train()
        return action[0]

    def distr_projection(self, next_distribution, rewards, dones):
        next_distr = next_distribution.data.cpu().numpy()
        rewards    = rewards.data.cpu().numpy()
        dones_mask = dones.cpu().numpy().astype(bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.n_atoms), dtype=np.float32)
        gamma      = self.gamma ** self.n_steps

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

    def step_training(self, state, action, reward, next_state, done, batch_size, per_memory_status):

        # Save experience in memory
        if per_memory_status:
            self.memory_per.per_add(state, action, reward, next_state, done)
        else:
            self.memory.replay_buffer_add(state, action, reward, next_state, done)

        LEARN_EVERY_STEP = 100
        self.t_step = self.t_step + 1

        if self.t_step % LEARN_EVERY_STEP == 0:
            self.learn_step(batch_size, per_memory_status)

    def learn_step(self, batch_size, per_memory_status):
        if per_memory_status:
            tree_idx, minibatch = self.memory_per.sample_experience(batch_size)
            states = np.zeros((batch_size, self.num_states))
            next_states = np.zeros((batch_size, self.num_states))
            actions, rewards, dones = [], [], []
            for i in range(batch_size):
                states[i] = minibatch[i][0]
                actions.append(minibatch[i][1])
                rewards.append(minibatch[i][2])
                next_states[i] = minibatch[i][3]
                dones.append(minibatch[i][4])
        else:
            # check, if enough samples are available in memory
            if self.memory.__len__() <= batch_size:
                return
            else:
                states, actions, rewards, next_states, dones = self.memory.sample_experience(batch_size)

        states  = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones   = np.array(dones)
        next_states = np.array(next_states)

        states  = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones   = torch.ByteTensor(dones)
        next_states = torch.FloatTensor(next_states)

        '''
        # this is from tutorial only
        # remeber here remove the softmax in the last critic's layer
        # ---------------------------- update critic ---------------------------- #
        crt_distr_v = self.critic.forward(states, actions)

        last_act_v   = self.actor_target.forward(next_states)
        last_distr_v = F.softmax(self.critic_target.forward(next_states, last_act_v), dim=1)

        proj_distr_v = self.distr_projection(last_distr_v, rewards, dones)
        prob_dist_v = -F.log_softmax(crt_distr_v, dim=1) * proj_distr_v
        critic_loss_v = prob_dist_v.sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss_v.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        actions_pred = self.actor.forward(states)
        crt_distr_v  = self.critic.forward(states, actions_pred)

        support_v = torch.arange(self.v_min, self.v_max + self.delta, self.delta)
        weights = F.softmax(crt_distr_v, dim=1) * support_v
        actor_loss_v = weights.sum(dim=1)
        actor_loss_v = - actor_loss_v.unsqueeze(dim=-1)  #  todo check this line
        actor_loss_v = actor_loss_v.mean()

        self.actor_optimizer.zero_grad()
        actor_loss_v.backward()
        self.actor_optimizer.step()


        # update the target networks using tao "soft updates"
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        '''
        # -------------------------------------------------------------------#
        # -------------------------------------------------------------------#

        # calculate the next Z distribution Z(s',a') --> Q_next_value
        next_actions = self.actor_target.forward(next_states)  # Note this is from actor-target
        next_Z_val   = self.critic_target.forward(next_states, next_actions.detach())

        # calculate the project target distribution Y --> Q_target
        proj_distr_v = self.distr_projection(next_Z_val, rewards, dones)
        Y = proj_distr_v  # target_z_projected

        # calculate the distribution prediction Z(s,a) --> Q_values
        Z_val = self.critic.forward(states, actions)  # this is a categorical distribution, the Z predicted
        # ----------------------------------- Calculate the loss ----- #
        # ------- calculate the critic loss
        BCE_loss = torch.nn.BCELoss(reduction='none')
        td_error = BCE_loss(Z_val, Y)
        td_error = td_error.mean(axis=1)
        critic_loss = td_error.mean()

        # ------- calculate the actor loss
        z_atoms = np.linspace(self.v_min, self.v_max, self.n_atoms)
        z_atoms = torch.from_numpy(z_atoms).float()
        actor_loss = self.critic.forward(states, self.actor.forward(states))
        actor_loss = actor_loss * z_atoms
        actor_loss = torch.sum(actor_loss, dim=1)
        actor_loss = -actor_loss.mean()

        # ---------Update priorities for PER
        if per_memory_status:
            td_error = td_error.detach().numpy().flatten()
            absolute_errors = np.abs(td_error)
            self.memory_per.batch_update(tree_idx, absolute_errors)

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

    EPISODES = 50000   # ---> T, total number of episodes
    batch_size = 64   # ---> M
    rollout_steps = 1  # ---> N, trajectory length
    gamma = 0.99
    # -------------------------------
    env = gym.make('Pendulum-v1')
    agent = D4PGAgent(env, gamma=gamma, n_steps=rollout_steps)
    # -------------------------------
    per_memory_status = False
    # -----------
    rewards = []
    avg_rewards = []
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()
            n_step_reward = 0
            for n in range(rollout_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                n_step_reward += reward * gamma ** n  # todo should use gamma here?
                if n == (rollout_steps - 1):
                    agent.step_training(state, action, n_step_reward, next_state, done, batch_size, per_memory_status)
                state = next_state
                episode_reward += reward

            if done:
                print(episode_reward, episode)
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

