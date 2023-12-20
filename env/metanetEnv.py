import gym
from gym import spaces
from env.metanet import *

class MetanetEnv(gym.Env):
    def __init__(self):
        # 定义动作空间和状态空间
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))   # 一维状态空间，并进行标准化处理[0,1]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,))  # 三维状态空间，分别为
        self.metanet = Metanet()

        # 初始化环境的内部状态
        self.state = None

    def reset(self):
        # 重置环境的状态
        self.state = self.np_random.uniform(low=0, high=1, size=(3,))
        return self.state

    def step(self, action):
        # 执行动作并返回下一个状态、奖励和是否终止的标志
        assert self.action_space.contains(action), "Invalid action"

        # 根据动作更新状态
        self.state += action

        # 计算奖励
        reward = self._calculate_reward()

        # 判断是否终止
        done = self._is_done()

        # 返回下一个状态、奖励和是否终止的标志
        return self.state, reward, done, {}

    def _calculate_reward(self):
        # 根据当前状态计算奖励
        # 这里只是一个示例，你可以根据自己的需求进行定义
        return sum(self.state)

    def _is_done(self):
        # 判断是否终止
        # 这里只是一个示例，你可以根据自己的需求进行定义
        return sum(self.state) > 100

    def render(self, mode='human'):
        # 可选的渲染函数，用于可视化环境
        pass

