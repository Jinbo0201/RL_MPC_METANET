# This is a sample Python script.
import pyomo.environ
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pyomo.environ as pyo
from pyomo.environ import *

def test_pyomo():
    # model = ConcreteModel()  # 定义模型
    # model.x = Var(within=NonNegativeReals)  # 声明决策变量 x
    # model.y = Var(within=NonNegativeReals)  # 声明决策变量 y
    # model.obj = Objective(expr=model.x + model.y, sense=minimize)  # 声明目标函数为 x+y, minimize 表示极小化
    # model.constrs = Constraint(expr=model.x + model.y <= 1)  # 添加约束 x+y <= 1
    # model.write('model.lp')  # 输出模型文件
    # model.pprint()  # 打印模型信息
    # opt = SolverFactory('ipopt')  # 指定Gurobi为求解器
    # solution = opt.solve(model)  # 调用求解器求解
    # print(solution)

    # model = ConcreteModel()
    # model.x = Var(initialize=0, bounds=(0, 1))
    # model.y = Var(initialize=0, bounds=(0, 1))
    # model.obj = Objective(expr=(model.x - 0.5) ** 2 + (model.y - 1) ** 2)
    # model.con1 = Constraint(expr=model.x + model.y >= 1)

    # model = pyo.ConcreteModel()
    # model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)
    # model.OBJ = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])
    # model.Constraint1 = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)
    # solver = pyo.SolverFactory('ipopt')
    # solver.solve(model)
    # model.display()



    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals, bounds=(1, 5))
    model.y = Var(within=NonNegativeReals, bounds=(1, 5))
    model.z = Var(within=NonNegativeReals, bounds=(1, 5))
    model.w = Var(within=NonNegativeReals, bounds=(1, 5))
    model.obj = Objective(expr=model.x * model.w * (model.x + model.y + model.z) + model.z, sense=minimize)
    model.con1 = Constraint(expr=model.x * model.y * model.z * model.w >= 25)
    model.con2 = Constraint(expr=model.x ** 2 + model.y ** 2 + model.z * pyomo.environ.exp(model.z) + model.w ** 2 <= 40)
    solver = SolverFactory('ipopt')
    solver.solve(model)
    print('Objective: ', model.obj())
    print('x: ', model.x())
    print('y: ', model.y())
    print('z: ', model.z())
    print('w: ', model.w())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    test_pyomo()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
