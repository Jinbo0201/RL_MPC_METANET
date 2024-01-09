import pyomo.environ as pyo
import matplotlib.pyplot as plt

NUM_SEGMENT = 3
ID_ONRAMP = 2 - 1
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

PWA_MAX = 1000
PWA_MIN = -1000
PWA_EPSILON = 0.01
PWA_X = 75.98

NP = 100  # 预测步长

# model = pyo.AbstractModel()
model = pyo.ConcreteModel()

# 状态变量
model.x_p = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_q = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_v = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
model.x_w_o = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
model.x_r_o = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
model.x_w_r = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
model.x_r_r = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
# 辅助变量
model.a_delta = pyo.Var(range(NUM_SEGMENT), range(NP), domain=pyo.Binary)
# 决策变量
model.u = pyo.Var(range(NP), domain=pyo.NonNegativeIntegers)
# 参数
model.p_d_o = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_d_r = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_p_e = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# 参数-初始状态
model.p_q = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_p = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_v = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_w_o = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
model.p_w_r = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)


# 目标函数
def obj_rule(model):
    return sum(L_SEGMENT * NUM_LINE * DELTA_T * model.x_p[id_segment, k] for id_segment in range(NUM_SEGMENT) for k in
               range(1, NP + 1)) \
        + 10 * sum(DELTA_T * model.x_w_o[k] for k in range(1, NP + 1)) \
        + 100 * sum(DELTA_T * model.x_w_r[k] for k in range(1, NP + 1))


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


def constraint_init_x_w_o(model):
    return model.x_w_o[0] == model.p_w_o


model.c_init_x_w_o = pyo.Constraint(rule=constraint_init_x_w_o)


def constraint_init_x_w_r(model):
    return model.x_w_r[0] == model.p_w_r


model.c_init_x_w_r = pyo.Constraint(rule=constraint_init_x_w_r)


def constraint_cal_q(model, id_segment, k):
    return model.x_q[id_segment, k] == NUM_LINE * model.x_p[id_segment, k] * model.x_v[id_segment, k]


model.c_cal_q = pyo.Constraint(range(NUM_SEGMENT), range(NP + 1), rule=constraint_cal_q)


def constraint_cal_p(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + DELTA_T / (L_SEGMENT * NUM_LINE) * (
                model.x_r_o[k - 1] - model.x_q[id_segment, k - 1])
    elif id_segment == ID_ONRAMP:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + DELTA_T / (L_SEGMENT * NUM_LINE) * (
                model.x_q[id_segment - 1, k - 1] - model.x_q[id_segment, k - 1] + model.x_r_r[k - 1])
    else:
        expr = model.x_p[id_segment, k] == model.x_p[id_segment, k - 1] + DELTA_T / (L_SEGMENT * NUM_LINE) * (
                model.x_q[id_segment - 1, k - 1] - model.x_q[id_segment, k - 1])
    return expr


model.c_cal_p = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_p)


# def constraint_cal_v(model, id_segment, k):
#     if id_segment == 0:
#         expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
#                 V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
#             id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
#                        model.x_v[id_segment, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
#                        TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
#                        model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
#                    id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
#     elif id_segment == NUM_SEGMENT - 1:
#         expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
#                 V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
#             id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
#                        model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
#                        TAU * L_SEGMENT) * (model.p_d - model.x_p[id_segment, k - 1]) / (
#                        model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
#                    id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
#     else:
#         expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
#                 V_MAX * (pyo.exp(-1 / A * (model.x_p[id_segment, k - 1] / DENSITY_MAX) ** A)) - model.x_v[
#             id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
#                        model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
#                        TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
#                        model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
#                    id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r[1, k - 1]
#     return expr
def constraint_cal_v(model, id_segment, k):
    if id_segment == 0:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                model.a_delta[id_segment, k - 1] * (102 - 1.3 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (5.58 - 0.031 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA)
    elif id_segment == NUM_SEGMENT - 1:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                model.a_delta[id_segment, k - 1] * (102 - 1.3 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (5.58 - 0.031 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.p_p_e - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA)
    elif id_segment == ID_ONRAMP:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                model.a_delta[id_segment, k - 1] * (102 - 1.3 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (5.58 - 0.031 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA) - MU * DELTA_T / (L_SEGMENT * NUM_LINE) * model.x_v[
                   id_segment, k - 1] / (model.x_p[id_segment, k - 1] + KAPPA) * model.x_r_r[k - 1]
    else:
        expr = model.x_v[id_segment, k] == model.x_v[id_segment, k - 1] + DELTA_T / TAU * (
                model.a_delta[id_segment, k - 1] * (102 - 1.3 * model.x_p[id_segment, k - 1]) + (
                1 - model.a_delta[id_segment, k - 1]) * (5.58 - 0.031 * model.x_p[id_segment, k - 1]) - model.x_v[
                    id_segment, k - 1]) + DELTA_T / L_SEGMENT * model.x_v[id_segment, k - 1] * (
                       model.x_v[id_segment - 1, k - 1] - model.x_v[id_segment, k - 1]) - ETA * DELTA_T / (
                       TAU * L_SEGMENT) * (model.x_p[id_segment + 1, k - 1] - model.x_p[id_segment, k - 1]) / (
                       model.x_p[id_segment, k - 1] + KAPPA)
    return expr


model.c_cal_v = pyo.Constraint(range(NUM_SEGMENT), range(1, NP + 1), rule=constraint_cal_v)


def constraint_aux_delta_1(model, id_segment, k):
    return model.x_p[id_segment, k] - PWA_X <= PWA_MAX * (1 - model.a_delta[id_segment, k])


model.c_aux_delta_1 = pyo.Constraint(range(NUM_SEGMENT), range(NP), rule=constraint_aux_delta_1)


def constraint_aux_delta_2(model, id_segment, k):
    return model.x_p[id_segment, k] - PWA_X >= PWA_EPSILON + (PWA_MIN - PWA_EPSILON) * model.a_delta[id_segment, k]


model.c_aux_delta_2 = pyo.Constraint(range(NUM_SEGMENT), range(NP), rule=constraint_aux_delta_2)


def constraint_cal_w_o(model, k):
    expr = model.x_w_o[k] == model.x_w_o[k - 1] + DELTA_T * (model.p_d_o - model.x_r_o[k - 1])
    return expr


model.c_cal_w_o = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_w_o)


def constraint_cal_w_r(model, k):
    expr = model.x_w_r[k] == model.x_w_r[k - 1] + DELTA_T * (model.p_d_r - model.x_r_r[k - 1])
    return expr


model.c_cal_w_r = pyo.Constraint(range(1, NP + 1), rule=constraint_cal_w_r)


def constraint_cal_r_o(model, k):
    return model.x_r_o[k] <= model.p_d_o + model.x_w_o[k] / DELTA_T


model.c_cal_r_o = pyo.Constraint(range(NP), rule=constraint_cal_r_o)


def constraint_cal_r_r(model, k):
    return model.x_r_r[k] <= model.p_d_r + model.x_w_r[k] / DELTA_T


model.c_cal_r_r = pyo.Constraint(range(NP), rule=constraint_cal_r_r)


def constraint_cal_r_o_capacity(model, k):
    return model.x_r_o[k] <= CAPACITY_ORIGIN * (DENSITY_MAX - model.x_p[0, k]) / (DENSITY_MAX - DENSITY_CRIT)


model.c_cal_r_o_capacity = pyo.Constraint(range(NP), rule=constraint_cal_r_o_capacity)


def constraint_cal_r_r_capacity(model, k):
    return model.x_r_r[k] <= CAPACITY_ONRAMP * (DENSITY_MAX - model.x_p[ID_ONRAMP, k]) / (DENSITY_MAX - DENSITY_CRIT)


model.c_cal_r_r_capacity = pyo.Constraint(range(NP), rule=constraint_cal_r_r_capacity)


def constraint_cal_r_o_c(model, k):
    return model.x_r_o[k] <= CAPACITY_ORIGIN


model.c_cal_r_o_c = pyo.Constraint(range(NP), rule=constraint_cal_r_o_c)


def constraint_cal_r_r_c(model, k):
    return model.x_r_r[k] <= CAPACITY_ONRAMP


model.c_cal_r_r_c = pyo.Constraint(range(NP), rule=constraint_cal_r_r_c)

model.pprint(verbose=True)

# # 参数
# model.p_d_o = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_d_r = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_p_e = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# # 参数-初始状态
# model.p_q = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_p = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_v = pyo.Param(range(NUM_SEGMENT), domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_w_o = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)
# model.p_w_r = pyo.Param(domain=pyo.NonNegativeReals, initialize=0, mutable=True)

model.p_d_o = 3000
model.p_d_r = 1500
model.p_p_e = 60

for id_segment in range(NUM_SEGMENT):
    model.p_q[id_segment] = 0
    model.p_p[id_segment] = 0
    model.p_v[id_segment] = FREE_V

model.p_w_o = 0
model.p_w_r = 0

solver = pyo.SolverFactory('ipopt')
results = solver.solve(model)

# # 状态变量
# model.x_p = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
# model.x_q = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
# model.x_v = pyo.Var(range(NUM_SEGMENT), range(NP + 1), domain=pyo.NonNegativeReals)
# model.x_w_o = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
# model.x_r_o = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
# model.x_w_r = pyo.Var(range(NP + 1), domain=pyo.NonNegativeReals)
# model.x_r_r = pyo.Var(range(NP), domain=pyo.NonNegativeReals)
# # 辅助变量
# model.a_delta = pyo.Var(range(NUM_SEGMENT), range(NP), domain=pyo.Binary)
# # 决策变量
# model.u = pyo.Var(range(NP), domain=pyo.NonNegativeIntegers)

print("Optimization status:", results.solver.status)
print("Optimal objective value:", pyo.value(model.obj))
print("Optimal variable values:")

r_list = []
w_list = []
r_list_o = []
w_list_o = []
v_list_0 = []
v_list_1 = []
v_list_2 = []
p_list_0 = []
p_list_1 = []
p_list_2 = []
q_list_0 = []
q_list_1 = []
q_list_2 = []
a_delta_list_0 = []
a_delta_list_1 = []
a_delta_list_2 = []

for k in range(NP):
    print(k, "w =", pyo.value(model.x_w_o[k]), pyo.value(model.x_w_r[k]))
    print(k, "r =", pyo.value(model.x_r_o[k]), pyo.value(model.x_r_r[k]))
    r_list.append(pyo.value(model.x_r_r[k]))
    w_list.append(pyo.value(model.x_w_r[k]))
    r_list_o.append(pyo.value(model.x_r_o[k]))
    w_list_o.append(pyo.value(model.x_w_o[k]))
    v_list_0.append(pyo.value(model.x_v[0, k]))
    v_list_1.append(pyo.value(model.x_v[1, k]))
    v_list_2.append(pyo.value(model.x_v[2, k]))
    p_list_0.append(pyo.value(model.x_p[0, k]))
    p_list_1.append(pyo.value(model.x_p[1, k]))
    p_list_2.append(pyo.value(model.x_p[2, k]))
    q_list_0.append(pyo.value(model.x_q[0, k]))
    q_list_1.append(pyo.value(model.x_q[1, k]))
    q_list_2.append(pyo.value(model.x_q[2, k]))
    a_delta_list_0.append(pyo.value(model.a_delta[0, k]))
    a_delta_list_1.append(pyo.value(model.a_delta[1, k]))
    a_delta_list_2.append(pyo.value(model.a_delta[2, k]))

fig, axes = plt.subplots(2, 3)
axes[0, 0].plot(r_list, label='r')
axes[0, 0].plot(r_list_o, label='o')
axes[0, 0].set_title('r_list')
axes[0, 0].legend()
axes[0, 1].plot(w_list, label='r')
axes[0, 1].plot(w_list_o, label='o')
axes[0, 1].set_title('queue_list')
axes[0, 1].legend()
axes[0, 2].plot(q_list_0, label='0')
axes[0, 2].plot(q_list_1, label='1')
axes[0, 2].plot(q_list_2, label='2')
axes[0, 2].set_title('flow_list')
axes[0, 2].legend()

axes[1, 0].plot(v_list_0, label='0')
axes[1, 0].plot(v_list_1, label='1')
axes[1, 0].plot(v_list_2, label='2')
axes[1, 0].set_title('v_list')
axes[1, 0].legend()
axes[1, 1].plot(p_list_0, label='0')
axes[1, 1].plot(p_list_1, label='1')
axes[1, 1].plot(p_list_2, label='2')
axes[1, 1].set_title('density_list')
axes[1, 1].legend()
axes[1, 2].plot(a_delta_list_0, label='0')
axes[1, 2].plot(a_delta_list_1, label='1')
axes[1, 2].plot(a_delta_list_2, label='2')
axes[1, 2].set_title('delta_list')
axes[1, 2].legend()
plt.tight_layout()
plt.show()
