import pyomo.environ as pyo

NUM_SEGMENT = 3
ID_ONRAMP = 3 - 1
DELTA_T = 10 / 3600  # 步长时间 h
FREE_V = 102  # 自由速度 km/h
L_SEGMENT = 1  # 路段长度 km
NUM_LINE = 2  # 车道数
TAU = 18 / 3600  # 速度计算参数 h
A = 1.867  # 速度计算参数 常量
DENSITY_CRIT = 33.5  # 速度计算参数 vel/km
ETA = 60  # 速度计算参数 km^2/h
KAPPA = 40  # 速度计算参数 vel/km
MU = 0.0122  # 速度计算参数 常量
CAPACITY_ORIGIN = 3500  # 入口最大容量 veh/h
CAPACITY_ONRAMP = 2000  # 上匝道最大容量 veh/h
DENSITY_MAX = 180  # 最大密度 veh/km

V_MAX = 120  # 最大速度，用于标准化
FLOW_MAX = 8040  # 最大流量用于标准化
QUEUE_LENGTH_ONRAMP_MAX = 2000  # 最大匝道排队长度用于标准化

NUM_ORIGIN = 2  # 路段车流入口数量
NP = 6  # 预测步长

# model = pyo.AbstractModel()
model = pyo.ConcreteModel()

# 状态变量
model.x_p = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_q = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_v = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_w = pyo.Var(range(NUM_ORIGIN), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_r = pyo.Var(range(NUM_ORIGIN), range(NP), domain=pyo.NonNegativeReals)
# 决策变量
model.u = pyo.Var(range(NP), domain=pyo.NonNegativeIntegers)
# 参数
model.p_o = pyo.Param(range(NUM_ORIGIN), domain=pyo.NonNegativeReals, initialize=0)
model.p_d = pyo.Param(domain=pyo.NonNegativeReals, initialize=0)
# 参数-初始状态
model.p_q = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0)
model.p_p = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0)
model.p_v = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0)
model.p_w = pyo.Param(range(NUM_ORIGIN), domain=pyo.NonNegativeReals, initialize=0)


# 目标函数
def obj_rule(model):
    return sum(L_SEGMENT * NUM_LINE * DELTA_T * model.x_p[id_segment, k] for id_segment in range(NUM_SEGMENT) for k in
               range(NP)) + sum(DELTA_T * model.x_w[id_origin, k] for id_origin in range(NUM_ORIGIN) for k in range(NP))


model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


# 约束条件
# 约束条件-初始状态
def constraint_init_x_p(model, id_segment):
    return model.x_p[id_segment, 0] == model.p_p[id_segment]


model.c_init_x_p = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_p)


def constraint_init_x_q(model, id_segment):
    return model.x_q[id_segment, 0] == model.p_q[id_segment]


model.c_init_x_q = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_q)


def constraint_init_x_v(model, id_segment):
    return model.x_v[id_segment, 0] == model.p_v[id_segment]


model.c_init_x_v = pyo.Constraint(range(NUM_SEGMENT), rule=constraint_init_x_v)


def constraint_init_x_w(model, id_origin):
    return model.x_w[id_origin, 0] == model.p_w[id_origin]


model.c_init_x_w = pyo.Constraint(range(NUM_ORIGIN), rule=constraint_init_x_w)


def constraint_cal_q(model, id_segment, k):
    return model.x_q[id_segment, k] == NUM_LINE * model.x_p[id_segment, k] * model.x_v[id_segment, k]


model.c_cal_q = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_q)


def constraint_cal_p(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + DELTA_T / (L_SEGMENT * NUM_LINE) * (
                model.x_r[0, k - 1] - model.x_q[id_segment, k - 1] + model.x_r[1, k - 1])
    else:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + DELTA_T / (L_SEGMENT * NUM_LINE) * (
                model.x_q[id_segment - 1, k - 1] - model.x_q[id_segment, k - 1] + model.x_r[1, k - 1])
    return expr


model.c_cal_p = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_p)


def constraint_cal_v(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
            id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
                   id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
    elif id_segment == NUM_SEGMENT - 1:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
            id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.p_d - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
                   id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
    else:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
            id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
                   id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
    return expr


model.c_cal_v = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_v)


def constraint_cal_w(model, id_origin, k):
    expr = model.x_w[id_origin, k] == model.x_w[id_origin, k - 1] + DELTA_T * (
            model.p_o[id_origin] - model.x_r[id_origin, k - 1])
    return expr


model.c_cal_w = pyo.Constraint(range(NUM_ORIGIN), range(1, NP + 1), rule=constraint_cal_w)


def constraint_cal_r(model, id_origin, k):
    if id_origin == 0:
        expr = model.x_r[id_origin, k] <= model.p_o[id_origin] + model.x_w[id_origin, k] / DELTA_T
    elif id_origin == 1:
        expr = model.x_r[id_origin, k] <= model.u[k] * model.p_o[id_origin] + model.x_w[id_origin, k] / DELTA_T
    return expr


model.c_cal_r = pyo.Constraint(range(NUM_ORIGIN), range(NP), rule=constraint_cal_r)


def constraint_cal_r_capacity(model, id_origin, k):
    if id_origin == 0:
        expr = model.x_r[id_origin, k] <= CAPACITY_ORIGIN * (DENSITY_MAX - model.x_p[0, k]) / (
                DENSITY_MAX - DENSITY_CRIT)
    elif id_origin == 1:
        expr = model.x_r[id_origin, k] <= CAPACITY_ONRAMP * (DENSITY_MAX - model.x_p[ID_ONRAMP, k]) / (
                DENSITY_MAX - DENSITY_CRIT)
    return expr


model.c_cal_r_capacity = pyo.Constraint(range(NUM_ORIGIN), range(NP), rule=constraint_cal_r_capacity)

model.pprint(verbose=True)

solver = pyo.SolverFactory('ipopt')
solver.solve(model)
print('Objective: ', model.obj())
