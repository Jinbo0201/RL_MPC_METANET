import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

FREE_V = 102  # 自由速度 km/h
A = 1.867  # 速度计算参数 常量
DENSITY_CRIT = 33.5  # 速度计算参数 vel/km
DENSITY_MAX = 180  # 最大密度 veh/km


def _get_Ve(density):
    return FREE_V * math.exp(-1 / A * (density / DENSITY_CRIT) ** A)


x_data = list(range(DENSITY_MAX + 1))
y_data = [_get_Ve(x) for x in x_data]
# print(y_data)


# def piecewise_linear(x, x0, y0):
#     return np.piecewise(x, [x < x0, x >= x0], [lambda x: y0 - (y0-102)/x0 * (x0 - x), lambda x: (0-y0)/(180-x0) * (x - x0) + y0])
# popt, pcov = curve_fit(piecewise_linear, x_data, y_data)
#
# # 打印拟合的参数
# print("拟合参数：", popt)
#
# # 绘制原始数据和拟合曲线
# plt.scatter(x_data, y_data, label='Data')
# plt.plot(x_data, piecewise_linear(x_data, *popt), 'r', label='Fit')
# plt.legend()
# plt.show()


def _get_Ve_p(x):
    if x < 75.89:
        return 3.21 - (3.21 - 102) / 75.89 * (75.89 - x)
    else:
        return (0 - 3.21) / (180 - 75.89) * (x - 75.89) + 3.21


def _get_Ve_PWA(density):
    return FREE_V


y_data_p = [_get_Ve_p(x) for x in x_data]

plt.figure()
plt.plot(x_data, y_data)
plt.plot(x_data, y_data_p)
plt.show()
