# -*- coding: utf-8 -*-
# time: 2023/5/7 15:39
# file: my_qubit_rotation.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com


import numpy as np

# 定义旋转门的参数
params = np.array([1.5, 1.6])
step_size = 0.1
num_iterations = 1000
tolerance = 1e-5


def rx_gate(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def ry_gate(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])


def initial_state():
    return np.array([1, 0])


def measure(state):
    prob = np.abs(state) ** 2
    if np.random.uniform(0, 1) < prob[0]:
        return 0
    else:
        return 1


def circuit(params):
    state = initial_state()
    state = np.dot(rx_gate(params[0]), state)
    state = np.dot(ry_gate(params[1]), state)
    return measure(state)


def cost(params):
    expectation = 0
    for i in range(100):
        expectation += circuit(params)
    expectation /= 100
    return expectation


def gradient(cost, params):
    grad = np.zeros_like(params)
    shift = np.zeros_like(params)
    for j in range(len(params)):
        shift[j] = np.pi / 2
        forward_params = params + shift
        backward_params = params - shift
        forward_cost = cost(forward_params)
        backward_cost = cost(backward_params)
        grad[j] = (forward_cost - backward_cost) / 2
        shift[j] = 0
    return grad


if __name__ == '__main__':
    # 初始化参数
    params = np.array([1.5, 1.6])
    prev_cost = cost(params)

    # 开始迭代
    for i in range(num_iterations):
        grad = gradient(cost, params)
        params -= step_size * grad
        curr_cost = cost(params)

        if np.abs(curr_cost - prev_cost) < tolerance:
            print("Converged after %d iterations." % i)
            break

        prev_cost = curr_cost

    print("Optimized parameters: ", params)
    print("Optimized cost: ", curr_cost)






