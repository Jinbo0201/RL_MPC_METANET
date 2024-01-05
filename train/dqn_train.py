import os.path
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from metanetGym.metanetEnv import MetanetEnv



# 定义神经网络模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义 DQN 算法
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 创建环境和智能体
env = MetanetEnv()
env_name = 'MetanetEnv'
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# 训练 DQN 智能体
return_list = []

episodes = 10
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay()

    print(f"Episode: {episode+1}, Reward: {total_reward}")
    return_list.append(total_reward)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
dqn_file_path = '../result'
dqn_file_name = os.path.join(dqn_file_path, f'dqn_model_{current_time}.pth')
torch.save(agent, dqn_file_name)
# dqn_model_para_dict = {
#     'batch_size': agent.batch_size,
#     'gamma': agent.gamma,
#     'epsilon ': agent.epsilon,
#     'epsilon_decay': agent.epsilon_decay,
#     'epsilon_min': agent.epsilon_min,
#     'others': None
# }
# dqn_model_para_json = json.dumps(dqn_model_para_dict, indent=4)
# dqn_para_file_name = os.path.join(dqn_file_path, f'dqn_model_{current_time}_para.json')
# with open(dqn_para_file_name, "w") as file:
#     file.write(dqn_model_para_json)

plt.figure()
plt.plot(return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))



# 使用训练好的智能体进行测试
action_list = []
reward_list = []
queue_list = []
episodes = 1
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        action = 3
        state, reward, done, _ = env.step(action)
        # env.render()
        total_reward += reward

        action_list.append(action)
        reward_list.append(reward)
        queue_list.append(state[-1])

    print(f"Test Episode: {episode+1}, Reward: {total_reward}")
    print(f"Test Episode: {episode + 1}, TTS: {0.1 * sum(action_list) - total_reward}")

plt.figure()
plt.plot(action_list)
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Action on {}'.format(env_name))

plt.figure()
plt.plot(reward_list)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward on {}'.format(env_name))

plt.figure()
plt.plot(queue_list)
plt.xlabel('Step')
plt.ylabel('Queue')
plt.title('Queue on {}'.format(env_name))

plt.show()