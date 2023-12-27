import math
import random


class Metanet(object):

    def __init__(self):
        # parameters
        self.NUM_SEGEMNT = 4
        self.ID_ONRAMP = 3 - 1
        self.DELTA_T = 10 / 3600  # 步长时间 h
        self.FREE_V = 102  # 自由速度 km/h
        self.L = 1  # 路段长度 km
        self.NUM_LINE = 2  # 车道数
        self.TAU = 18 / 3600  # 速度计算参数 h
        self.A = 1.867  # 速度计算参数 常量
        self.DENSITY_CRIT = 33.5  # 速度计算参数 vel/km
        self.ETA = 60  # 速度计算参数 km^2/h
        self.KAPPA = 40  # 速度计算参数 vel/km
        self.MU = 0.0122  # 速度计算参数 常量
        self.CAPACITY_ORIGIN = 3500  # 入口最大容量 veh/h
        self.CAPACITY_ONRAMP = 2000  # 上匝道最大容量 veh/h
        self.DENSITY_MAX = 180  # 最大密度 veh/km

        self.V_MAX = 120    # 最大速度，用于标准化
        self.FLOW_MAX = 8040    # 最大流量用于标准化
        self.QUEUE_LENGTH_ONRAMP_MAX = 2000    # 最大匝道排队长度用于标准化

        # self.RANDOM_DEMAND_ORIGN_CYCLE = random.choice([0.5, 1, 1.5])
        # self.RANDOM_DEMAND_ORIGN_MAX = random.randint(2500, 3500)
        # self.RANDOM_DEMAND_ORIGN_MIN = random.randint(500, 1500)
        # self.RANDOM_DEMAND_ONRAMP_CYCLE = random.choice([0.5, 1, 1.5])
        # self.RANDOM_DEMAND_ONRAMP_MAX = random.randint(1000, 2000)
        # self.RANDOM_DEMAND_ONRAMP_MIN = random.randint(500, 1000)
        # self.RANDOM_DOWNSTREAM_DENSITY_CYCLE = random.choice([0.5, 1, 1.5, 2])
        # self.RANDOM_DOWNSTREAM_DENSITY_MAX = random.randint(50, 70)
        # self.RANDOM_DOWNSTREAM_DENSITY_MIN = random.randint(10, 30)
        self.RANDOM_DEMAND_ORIGN_CYCLE = 1
        self.RANDOM_DEMAND_ORIGN_MAX = 2500
        self.RANDOM_DEMAND_ORIGN_MIN = 1000
        self.RANDOM_DEMAND_ONRAMP_CYCLE = 1
        self.RANDOM_DEMAND_ONRAMP_MAX = 1500
        self.RANDOM_DEMAND_ONRAMP_MIN = 750
        self.RANDOM_DOWNSTREAM_DENSITY_CYCLE = 1
        self.RANDOM_DOWNSTREAM_DENSITY_MAX = 60
        self.RANDOM_DOWNSTREAM_DENSITY_MIN = 20
        # states
        self.state_density = [0] * self.NUM_SEGEMNT
        self.state_flow = [0] * self.NUM_SEGEMNT
        self.state_v = [self.FREE_V] * self.NUM_SEGEMNT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * self.NUM_SEGEMNT
        # inputs
        self.input_demand_origin = 0  # 入口处的需求，即流量
        self.input_demand_onramp = 0  # 上匝道的需求，即流量
        self.input_downsteam_density = 0  # 出口处的密度
        # actions
        self.action = 1
        # step
        self.step_id = 0

    # 初始化状态量
    def init_state(self):
        # states
        self.state_density = [0] * self.NUM_SEGEMNT
        self.state_flow = [0] * self.NUM_SEGEMNT
        self.state_v = [self.FREE_V] * self.NUM_SEGEMNT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * self.NUM_SEGEMNT
        # inputs
        self.input_demand_origin = 0  # 入口处的需求，即流量
        self.input_demand_onramp = 0  # 上匝道的需求，即流量
        self.input_downsteam_density = 0  # 出口处的密度
        # actions
        self.action = 1
        # step
        self.step_id = 0

    # 步进仿真
    def step_state(self, action):
        self.action = action

        self._cal_demand_origin()
        self._cal_demand_onramp()
        self._cal_downstream_density()

        self._cal_flow_onramp()
        self._cal_state_flow()
        self._cal_state_v()
        self._cal_state_density()

        self._cal_queue_length_onramp()
        self._cal_queue_length_origin()

        self.step_id += 1


    # 获取状态量
    def get_state(self):
        state_dict = {}
        state_dict['density'] = self.state_density
        state_dict['flow'] = self.state_flow
        state_dict['v'] = self.state_v
        state_dict['queue_length_origin'] = self.state_queue_length_origin
        state_dict['queue_length_onramp'] = self.state_queue_length_onramp
        state_dict['flow_onramp'] = self.state_flow_onramp
        return state_dict


    def _cal_state_flow(self):
        for id_segment in range(self.NUM_SEGEMNT):
            self.state_flow[id_segment] = self.NUM_LINE * self.state_density[id_segment] * self.state_v[id_segment]

    def _cal_state_v(self):
        for id_segment in range(self.NUM_SEGEMNT):
            if id_segment == 0:
                self.state_v[id_segment] = self.state_v[id_segment] + self.DELTA_T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.DELTA_T / self.L * (
                                                   self.state_v[id_segment] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.ETA * self.DELTA_T) / (
                                                       self.TAU * self.L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA) - (
                                                   self.MU * self.DELTA_T * self.state_flow_onramp[id_segment] *
                                                   self.state_v[id_segment]) / (
                                                   self.L * self.NUM_LINE * (
                                                       self.state_density[id_segment] + self.KAPPA))
            elif id_segment == self.NUM_SEGEMNT - 1:
                self.state_v[id_segment] = self.state_v[id_segment] + self.DELTA_T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.DELTA_T / self.L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.ETA * self.DELTA_T) / (
                                                       self.TAU * self.L) * (
                                                   self._get_destination_flow_max() - self.state_density[id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA) - (
                                                   self.MU * self.DELTA_T * self.state_flow_onramp[id_segment] *
                                                   self.state_v[id_segment]) / (
                                                   self.L * self.NUM_LINE * (
                                                       self.state_density[id_segment] + self.KAPPA))
            else:
                self.state_v[id_segment] = self.state_v[id_segment] + self.DELTA_T / self.TAU * (self._get_Ve(
                    self.state_density[id_segment]) - self.state_v[id_segment]) + self.DELTA_T / self.L * (
                                                   self.state_v[id_segment - 1] - self.state_v[id_segment]) * \
                                           self.state_v[id_segment] - (self.ETA * self.DELTA_T) / (
                                                       self.TAU * self.L) * (
                                                   self.state_density[id_segment + 1] - self.state_density[
                                               id_segment]) / (
                                                   self.state_density[id_segment] + self.KAPPA) - (
                                                   self.MU * self.DELTA_T * self.state_flow_onramp[id_segment] *
                                                   self.state_v[id_segment]) / (
                                                   self.L * self.NUM_LINE * (
                                                       self.state_density[id_segment] + self.KAPPA))

    def _cal_state_density(self):
        for id_segment in range(self.NUM_SEGEMNT):
            if id_segment == 0:
                self.state_density[id_segment] = self.state_density[id_segment] + self.DELTA_T / (
                        self.L * self.NUM_LINE) * (self._get_flow_origin_min() - self.state_flow[id_segment] +
                                                   self.state_flow_onramp[id_segment])
            else:
                self.state_density[id_segment] = self.state_density[id_segment] + self.DELTA_T / (
                        self.L * self.NUM_LINE) * (self.state_flow[id_segment - 1] - self.state_flow[id_segment] +
                                                   self.state_flow_onramp[id_segment])

    def _get_Ve(self, density):
        return self.FREE_V * math.exp(-1 / self.A * (density / self.DENSITY_CRIT) ** self.A)


    def _cal_queue_length_origin(self):
        self.state_queue_length_origin = self.state_queue_length_origin + self.DELTA_T * (
                    self.input_demand_origin - self._get_flow_origin_min())

    def _cal_queue_length_onramp(self):
        self.state_queue_length_onramp = self.state_queue_length_onramp + self.DELTA_T * (
                    self.input_demand_onramp - self.state_flow_onramp[self.ID_ONRAMP])

    def _get_flow_origin_min(self):
        value = min(self.input_demand_origin + self.state_queue_length_origin / self.DELTA_T, self.CAPACITY_ORIGIN,
                    self.CAPACITY_ORIGIN * (self.DENSITY_MAX - self.state_density[0]) / (self.DENSITY_MAX - self.DENSITY_CRIT))
        return value

    def _get_flow_onramp_min(self):
        value = min(self.input_demand_onramp + self.state_queue_length_onramp / self.DELTA_T, self.CAPACITY_ONRAMP,
                    self.CAPACITY_ONRAMP * (self.DENSITY_MAX - self.state_density[self.ID_ONRAMP]) / (
                                self.DENSITY_MAX - self.DENSITY_CRIT))
        return value

    def _cal_flow_onramp(self):
        self.state_flow_onramp[self.ID_ONRAMP] = self.action * self._get_flow_onramp_min()

    def _get_destination_flow_max(self):
        value = max(min(self.state_density[self.NUM_SEGEMNT-1], self.DENSITY_CRIT), self.input_downsteam_density)
        return value

    def _cal_demand_origin(self):

        delta = self.RANDOM_DEMAND_ORIGN_CYCLE / 4
        value = (self.step_id * self.DELTA_T) % self.RANDOM_DEMAND_ORIGN_CYCLE

        if value < delta:
            demand_origin = self.RANDOM_DEMAND_ORIGN_MIN
        elif value < delta * 2:
            demand_origin = self.RANDOM_DEMAND_ORIGN_MIN + (self.RANDOM_DEMAND_ORIGN_MAX - self.RANDOM_DEMAND_ORIGN_MIN) / delta * (value - delta)
        elif value < delta * 3:
            demand_origin = self.RANDOM_DEMAND_ORIGN_MAX
        else:
            demand_origin = self.RANDOM_DEMAND_ORIGN_MIN
        self.input_demand_origin = demand_origin

    def _cal_demand_onramp(self):
        delta = self.RANDOM_DEMAND_ONRAMP_CYCLE / 4
        value = (self.step_id * self.DELTA_T) % self.RANDOM_DEMAND_ONRAMP_CYCLE

        if value < delta:
            demand_onramp = self.RANDOM_DEMAND_ONRAMP_MIN
        elif value < delta * 2:
            demand_onramp = self.RANDOM_DEMAND_ONRAMP_MIN + (
                        self.RANDOM_DEMAND_ONRAMP_MAX - self.RANDOM_DEMAND_ONRAMP_MIN) / delta * (value - delta)
        elif value < delta * 3:
            demand_onramp = self.RANDOM_DEMAND_ONRAMP_MAX
        else:
            demand_onramp = self.RANDOM_DEMAND_ONRAMP_MIN
        self.input_demand_onramp = demand_onramp

    def _cal_downstream_density(self):
        delta = self.RANDOM_DOWNSTREAM_DENSITY_CYCLE / 4
        value = (self.step_id * self.DELTA_T) % self.RANDOM_DOWNSTREAM_DENSITY_CYCLE
        if value < delta:
            downstream_density = self.RANDOM_DOWNSTREAM_DENSITY_MIN
        elif value < delta * 2:
            downstream_density = self.RANDOM_DOWNSTREAM_DENSITY_MIN + (
                    self.RANDOM_DOWNSTREAM_DENSITY_MAX - self.RANDOM_DOWNSTREAM_DENSITY_MIN) / delta * (value - delta)
        elif value < delta * 3:
            downstream_density = self.RANDOM_DOWNSTREAM_DENSITY_MAX
        else:
            downstream_density = self.RANDOM_DOWNSTREAM_DENSITY_MIN
        self.input_downsteam_density = downstream_density
