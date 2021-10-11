# 使用HashER的主程序，调用了random_dict.py依赖
import gym
import random
import numpy as np
import numpy.random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import matplotlib.pyplot as plt

import time

from random_dict import RandomDict
from collections import deque


# 经验回放池
class ReplayMemory(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.size_now = 0
        # self.data = np.zeros(capacity, dtype=object)

        self.pair_to_indices_dict = RandomDict()  # 随机的键值对

        # TODO 确定observation的类型 dtype
        self.observations = np.zeros(capacity, dtype=object)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.observations_ = np.zeros(capacity, dtype=object)
        self.dones = np.zeros(capacity, dtype=np.float32)

    # 存数据 agent.memory.push(observation, action, reward, observation_, done)
    def push(self, observation, action, reward, observation_, done):
        p = self.data_pointer

        if self._is_full():
            old_pair = make_pair(self.observations[p], self.actions[p])
            old_index_deque = self.pair_to_indices_dict[old_pair]  # 随机取出
            old_index_deque.popleft()
            if not old_index_deque:
                self.pair_to_indices_dict.pop(old_pair)

        # 对于新的输入更新index
        new_pair = make_pair(observation, action)
        # print("new_pair:", new_pair)
        if new_pair not in self.pair_to_indices_dict:
            self.pair_to_indices_dict[new_pair] = deque()
        self.pair_to_indices_dict[new_pair].append(p)

        # 保存数据
        self.observations[p], self.actions[p], self.rewards[p], self.observations_[p], self.dones[
            p] = observation, action, reward, observation_, done
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size_now = min((self.size_now + 1), self.capacity)

    # 取数据
    def sample(self, n):
        # idx = random.sample(range(self.total_num_of_samples), n)
        # return self.data[idx]

        i = np.asarray([self._sample_index() for _ in range(n)])

        observations = self.observations[i]
        actions = self.actions[i]
        rewards = self.rewards[i]
        dones = self.dones[i]
        observations_ = self.observations_[i]

        return observations, actions, rewards, observations_, dones

    def _is_full(self):
        return self.size_now == self.capacity

    def _sample_index(self):
        index_deque = self.pair_to_indices_dict.random_value()
        x = random.choice(index_deque)
        if (x - self.data_pointer) % self.size_now >= self.capacity - 1:
            x = np.random.randint(self.size_now - 1)
        return (self.data_pointer + x) % self.size_now


def make_pair(observation, action):
    return hash(observation.tostring()), action.item()


# 神经网络
class DQN(torch.nn.Module):

    def __init__(self, in_dim, n_hidden, out_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden)
        self.layer2 = nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x


class Agent(object):
    def __init__(self):
        self.batch_size = 32
        self.gamma = 0.9

        s_dim = 4
        a_dim = 2

        self._policy_net = DQN(s_dim, 256, a_dim)
        self._target_net = DQN(s_dim, 256, a_dim)
        self._optim = optim.Adam(self._policy_net.parameters(), lr=0.0003)
        self.memory = ReplayMemory(1000)

        self._steps_done = 0
        self._eps_threshold = 1.

    # 探索策略
    def select_action(self, ob):
        self._steps_done += 1
        sample = random.random()
        if self._eps_threshold > 0.05:
            self._eps_threshold -= 0.0001
        if sample > self._eps_threshold:
            ob = torch.tensor(ob, dtype=torch.float).unsqueeze(0)
            return self._policy_net(ob).max(1)[1].detach().numpy()
        else:
            return torch.randint(0, 2, (1,)).detach().numpy()

    # 测试策略
    def select_test_action(self, ob):
        sample = random.random()
        if sample > 0.001:
            ob = torch.tensor(ob, dtype=torch.float).unsqueeze(0)
            return self._policy_net(ob).max(1)[1].detach().numpy()
        else:
            return torch.randint(0, 2, (1,)).detach().numpy()

    # 训练
    def learn(self):

        if self._steps_done < self.batch_size * 10:
            return

        if self._steps_done % 100 == 0:
            torch.save(self._policy_net.state_dict(), './data/model_weights_dqn.pth')
            self._target_net.load_state_dict(torch.load('./data/model_weights_dqn.pth'))

        s0, a0, r0, s1, d0 = self.memory.sample(self.batch_size)

        s0 = np.vstack(s0)
        s1 = np.vstack(s1)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.int64).view(-1, 1)
        r0 = torch.tensor(r0, dtype=torch.float).view(-1, 1)
        s1 = torch.tensor(s1, dtype=torch.float)
        d0 = torch.tensor(d0, dtype=torch.float).view(-1, 1)

        # print("a0:", type(a0), "\nao:", a0)

        y_pred = self._policy_net(s0).gather(1, a0)
        y_true = r0 + (1. - d0) * self.gamma * (self._target_net(s1).max(1)[0].view(-1, 1))

        loss = f.mse_loss(y_pred, y_true)
        self._policy_net.zero_grad()
        loss.backward()
        self._optim.step()

    def save(self):
        torch.save(self._policy_net.state_dict(), './data/model_weights_dqn.pth')


# 测试分数
def score_test(agent_t, env_t):
    observation_t = env_t.reset()
    score = 0
    while True:
        score += 1
        action_t = agent_t.select_test_action(observation_t)
        observation__t, reward_t, done_t, _ = env.step(int(action_t))
        observation_t = observation__t
        if score >= target_score or done_t is True:
            break
    return score


# 设置随机数种子
def setup_seed(environment, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    environment.seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)


time_start = time.time()
ENV_NAME = 'CartPole-v0'
SEED = 123456
target_score = 500

env = gym.make(ENV_NAME).unwrapped
setup_seed(env, SEED)
agent = Agent()

average_score = 0
average_scores = []
episode_scores = []
episode = 0
done = True
observation = None
step = 0

# 主循环
while True:

    if done or step >= target_score:

        episode_score = score_test(agent, env)
        average_score = 0.05 * episode_score + 0.95 * average_score
        episode_scores.append(episode_score)
        average_scores.append(average_score)

        if episode % 10 == 0:
            print('Episode {}\tLast score: {:.2f}\tAverage score: {:.2f}'.format(
                episode, episode_score, average_score))

        if average_score > 0.97 * target_score:
            time_end = time.time()
            time_c = time_end - time_start
            agent.save()
            print("Solved! Average score is now {:.2f} , cost {:.2f}s and "
                  "the total episodes is {}!".format(average_score, time_c, episode))
            break

        observation = env.reset()
        episode += 1
        episode_score = 0
        step = 0

    step += 1

    action = agent.select_action(observation)
    observation_, reward, done, _ = env.step(int(action))
    reward = np.array([reward])
    agent.memory.push(observation, action, reward, observation_, done)
    agent.learn()
    observation = observation_

# 绘图
plt.title('DQN cart-pole with Hash ER')
plt.xlabel('Episode')
plt.ylabel('score')
plt.plot(average_scores, 'r-', label='average_score')
plt.plot(episode_scores, 'b.', label='episode_score')
plt.legend()
plt.show()
