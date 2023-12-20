class Metanet(object):

    def __init__(self):
        # parameters
        self.NUM_SEGEMNT = 4
        self.ID_OFFRAMP = 2
        self.ID_ONRAMP = 4
        # states
        self.state_density = [None] * self.NUM_SEGEMNT
        self.state_flow = [None] * self.NUM_SEGEMNT
        self.state_v = [None] * self.NUM_SEGEMNT
        self.state_queue_length = [None] * 2    #包括起点和上行匝道
        # inputs
        self.input_demand = [None] * 2    #包括起点和上行匝道
        self.input_downstream_density = None    #出口位置的
        # actions
        self.action = None

    #初始化状态量
    def init_state(self):
        # TODO: init the state
        pass

    #步进仿真
    def step_state(self):
        # TODO: realize the state calculation
        pass


    #获取状态量
    def get_state(self):
        # TODO: return the state
        pass



