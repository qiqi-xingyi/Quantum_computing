# -*- coding: utf-8 -*-
# time: 2023/5/9 19:39
# file: test_qubit_rotation2.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np

# 定义旋转门的参数
params = np.array([1.8, 1.62])

# 定义旋转门操作的矩阵
rx_matrix = np.array([[np.cos(params[0] / 2), -1j * np.sin(params[0] / 2)],
                      [-1j * np.sin(params[0] / 2), np.cos(params[0] / 2)]])
ry_matrix = np.array([[np.cos(params[1] / 2), -np.sin(params[1] / 2)],
                      [np.sin(params[1] / 2), np.cos(params[1] / 2)]])

# 定义Pauli-Z算符的矩阵
pauli_z_matrix = np.array([[1, 0], [0, -1]])

# 定义初始量子态
initial_state = np.array([1, 0])

# 应用旋转门操作
state_after_rx = np.dot(rx_matrix, initial_state)
state_after_ry = np.dot(ry_matrix, state_after_rx)

# 计算测量量子比特的期望值
expectation = np.dot(np.dot(state_after_ry.conj().T, pauli_z_matrix), state_after_ry)

print("Expectation value: ", expectation)

# 使用梯度下降优化器来最小化 cost 函数
def cost(x):
    rx_matrix = np.array([[np.cos(x[0] / 2), -1j * np.sin(x[0] / 2)],
                          [-1j * np.sin(x[0] / 2), np.cos(x[0] / 2)]])
    ry_matrix = np.array([[np.cos(x[1] / 2), -np.sin(x[1] / 2)],
                          [np.sin(x[1] / 2), np.cos(x[1] / 2)]])

    state_after_rx = np.dot(rx_matrix, initial_state)
    state_after_ry = np.dot(ry_matrix, state_after_rx)

    return np.dot(np.dot(state_after_ry.conj().T, pauli_z_matrix), state_after_ry)


if __name__ == '__main__':

    # 定义初始参数值
    init_params = np.array([1.2 , 1.6])

    # 设置优化器的参数
    stepsize = 0.004
    steps = 2000

    # 开始梯度下降优化
    params = init_params
    for i in range(steps):
        gradient = np.zeros_like(params)
        for j in range(len(params)):
            # 计算正向和反向的 cost 值
            shift = np.zeros_like(params)
            shift[j] = np.pi/2
            forward_params = params + shift
            backward_params = params - shift
            forward_cost = cost(forward_params)
            backward_cost = cost(backward_params)
            gradient[j] = (forward_cost - backward_cost) / 2
        # 更新参数值
        params -= stepsize * gradient

        if (i + 1) % 2 == 0:
            print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

    print("Optimized rotation angles: {}".format(params))

