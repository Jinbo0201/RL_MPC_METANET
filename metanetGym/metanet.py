import math


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
        self.FLOW_CRIT = 33.5  # 速度计算参数 vel/km
        self.ETA = 60  # 速度计算参数 km^2/h
        self.KAPPA = 40  # 速度计算参数 vel/km
        self.MU = 0.0122  # 速度计算参数 常量
        self.CAPACITY_ORIGIN = 3500  # 入口最大容量 veh/h
        self.CAPACITY_ONRAMP = 2000  # 上匝道最大容量 veh/h
        self.FLOW_MAX = 1800  # 最大流量 veh/h
        # states
        self.state_density = [0] * self.NUM_SEGEMNT
        self.state_flow = [0] * self.NUM_SEGEMNT
        self.state_v = [0] * self.NUM_SEGEMNT
        self.state_queue_length_origin = 0  # 入口处的队伍长度
        self.state_queue_length_onramp = 0  # 上匝道的队伍长度
        self.state_flow_onramp = [0] * self.NUM_SEGEMNT
        # inputs
        self.input_demand_origin = 0  # 入口处的需求，即流量
        self.input_demand_onramp = 0  # 上匝道的需求，即流量
        self.input_downsteam_density = 0  # 出口处的密度
        # actions
        self.action = 0
        # step
        self.step = 0

    # 初始化状态量
    def init_state(self):
        self.step = 0
        self.state_density = [0] * self.NUM_SEGEMNT
        self.state_v = [self.FREE_V] * self.NUM_SEGEMNT
        self.state_queue_length = [0] * self.NUM_SEGEMNT
        self.state_flow_onramp[self.ID_ONRAMP] = self.action

    # 步进仿真
    def step_state(self):
        self._cal_demand_origin()
        self._cal_demand_onramp()
        self._cal_downstream_density()

        self._cal_queue_length_onramp()
        self._cal_queue_length_origin()

        self._cal_state_flow()
        self._cal_state_v()
        self._cal_state_density()

        self.step += 1


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
                                                   self.input_downsteam_density - self.state_density[id_segment]) / (
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
        return self.FREE_V * math.exp(-1 / self.A * (density / self.FLOW_CRIT) ** self.A)

    def _get_flow_origin_min(self):
        value = min(self.input_demand_origin + self.state_queue_length_origin / self.DELTA_T, self.CAPACITY_ORIGIN,
                    self.CAPACITY_ORIGIN * (self.FLOW_MAX - self.state_flow[0]) / (self.FLOW_MAX - self.FLOW_CRIT))
        return value

    def _get_flow_onramp_min(self):
        value = min(self.input_demand_onramp + self.state_queue_length_onramp / self.DELTA_T, self.CAPACITY_ONRAMP,
                    self.CAPACITY_ONRAMP * (self.FLOW_MAX - self.state_flow[self.ID_ONRAMP]) / (
                                self.FLOW_MAX - self.FLOW_CRIT))
        return value

    def _cal_queue_length_origin(self):
        self.state_queue_length_origin = self.state_queue_length_origin + self.DELTA_T * (
                    self.input_demand_origin - self.state_flow_onramp[0])

    def _cal_queue_length_onramp(self):
        self.state_queue_length_onramp = self.state_queue_length_onramp + self.DELTA_T * (
                    self.input_demand_onramp - self._get_flow_origin_min())



    # 获取状态量
    def get_state(self):
        # TODO: return the state
        print(self.step, ':', self.state_flow)

    def _cal_demand_origin(self):
        demand_origin = 0
        if self.step < 20 * 6:
            demand_origin = 1000
        elif self.step < 60 * 6:
            demand_origin = 3000
        elif self.step < 120 * 6:
            demand_origin = 1000
        elif self.step < 140 * 6:
            demand_origin = 3000
        elif self.step < 200 * 6:
            demand_origin = 1000
        elif self.step <= 240 * 6:
            demand_origin = 3000
        else:
            print('get demand_origin out of range')
        self.input_demand_origin = demand_origin

    def _cal_demand_onramp(self):
        demand_onramp = 0
        if self.step < 20 * 6:
            demand_onramp = 1000
        elif self.step < 60 * 6:
            demand_onramp = 3000
        elif self.step < 120 * 6:
            demand_onramp = 1000
        elif self.step < 140 * 6:
            demand_onramp = 3000
        elif self.step < 200 * 6:
            demand_onramp = 1000
        elif self.step <= 240 * 6:
            demand_onramp = 3000
        else:
            print('get demand_onramp out of range')
        self.input_demand_onramp = demand_onramp

    def _cal_downstream_density(self):
        downstream_density = 0
        if self.step < 20 * 6:
            downstream_density = 20
        elif self.step < 60 * 6:
            downstream_density = 60
        elif self.step < 120 * 6:
            downstream_density = 20
        elif self.step < 140 * 6:
            downstream_density = 60
        elif self.step < 200 * 6:
            downstream_density = 20
        elif self.step <= 240 * 6:
            downstream_density = 60
        else:
            print('get downstream_density out of range')
        return downstream_density
