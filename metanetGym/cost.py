from metanetGym.metanetEnv import *

TIME_LENGTH_STEP = 5 / 60  # 每个控制步长的时长，小时

class CostMetanet(object):

    def __init__(self):
        self.env_cost = MetanetEnv()
        # self.env.metanet.DELTA_T = DELTA_TIME

    def cal_cost(self, action_list, state_begin, step_id_begin):
        self.env_cost = MetanetEnv()
        self.env_cost.reset()
        self.env_cost.set_state(state_begin, step_id_begin)
        total_reward = 0

        for action in action_list:
            for id_step in range(int(TIME_LENGTH_STEP/self.env_cost.metanet.DELTA_T)):
                observation, reward, done, _ = self.env_cost.step(action)
                total_reward += reward
        # self.env_cost.render()
        return total_reward


