import gym
from gym import spaces
from metanetGym.metanet import *

VALUE_ACTION2ACTION = 0.25


class MetanetEnv(gym.Env):
    def __init__(self):
        # 定义动作空间和状态空间
        self.action_space = spaces.Discrete(5)  # 离散动作，分别对应[0，0.25，0.5，0.75，1]，因此需要乘以VALUE_ACTION2ACTION = 0.25
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,))  # 三维状态空间，分别为横坐标为flow、speed，纵坐标为segment_1、2、3
        # 定义METANET
        self.metanet = Metanet()
        # 初始化环境的内部状态
        self.action = None
        self.state = None
        self.observation = None
        self.reward = None

    def reset(self):
        self.metanet.init_state()
        # 重置环境的状态
        self.action = None
        self.state = self.metanet.get_state()
        self.observation = self._get_observation()
        self.reward = None
        return self.observation

    def set_state(self, state, step_id):
        self.metanet.state_density = state['density']
        self.metanet.state_flow = state['flow']
        self.metanet.state_v = state['v']
        self.metanet.state_queue_length_origin = state['queue_length_origin']
        self.metanet.state_queue_length_onramp = state['queue_length_onramp']
        self.metanet.state_flow_onramp = state['flow_onramp']
        self.metanet.step_id = step_id

    def step(self, action):
        # 执行动作并返回下一个状态、奖励和是否终止的标志
        # 判断action是否在范围内
        self.action = action
        assert self.action_space.contains(self.action), "Invalid action"
        # 输入动作，根据动作步进仿真
        self.metanet.step_state(VALUE_ACTION2ACTION * self.action)
        # 获取状态量
        self.state = self.metanet.get_state()
        self.observation = self._get_observation()
        # 计算奖励
        self.reward = self._calculate_reward()
        # 判断是否终止
        done = self._is_done()
        # 返回下一个状态、奖励和是否终止的标志
        return self.observation, self.reward, done, {}

    def _calculate_reward(self):
        # 根据当前状态计算奖励
        reward_online = self.metanet.DELTA_T * sum(
            self.state['density']) * self.metanet.L_SEGMENT * self.metanet.NUM_LINE
        reward_queue = self.metanet.DELTA_T * (self.state['queue_length_origin'] + self.state['queue_length_onramp'])
        # reward_action = 0.1 * self.action
        reward_action = 0
        return reward_online + reward_queue + reward_action

    def _is_done(self):
        # 判断是否终止
        # 训练6个小时
        return self.metanet.step_id * self.metanet.DELTA_T > 6

    def _get_observation(self):
        observation = [
            self.state['flow'][0] / self.metanet.FLOW_MAX,
            self.state['flow'][1] / self.metanet.FLOW_MAX,
            self.state['flow'][2] / self.metanet.FLOW_MAX,
            self.state['v'][0] / self.metanet.V_MAX,
            self.state['v'][1] / self.metanet.V_MAX,
            self.state['v'][2] / self.metanet.V_MAX,
            self.state['queue_length_onramp'] / self.metanet.QUEUE_LENGTH_ONRAMP_MAX
        ]
        return observation

    def render(self):
        # 可选的渲染函数，用于可视化环境
        print('step:', self.metanet.step_id, ', action:', self.action, ', reward:', self.reward)
        print('state', self.state)
