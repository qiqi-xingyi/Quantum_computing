# -*- coding: utf-8 -*-
# time: 2023/5/16 17:37
# file: Tensor-network.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt



BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
#生成图片的点阵

j = 1
plt.figure(figsize=[3, 3])
for i in BAS:
    plt.subplot(2, 2, j)
    j += 1
    plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
    plt.xticks([])
    plt.yticks([])


#定义操作Rx和Ry
def Rx(theta):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def Ry(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])


def quit_rotation(angle, wire):

    global state

    #初始量子态
    state = np.array([1, 0], dtype=np.complex128)

    # cos = np.cos(angle)
    # sin = np.sin(angle)
    #
    # Ry = np.array([[cos, -sin], [sin, cos]])

    state = np.dot(Rx(angle) , state)

    state = np.dot(Ry(angle) , state)


def apply_cnot(wires):
    global state

    # state = np.reshape(state, (1, 4))

    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

    state = np.dot(CNOT, state)


def block(weights, wires):
    # apply_ry(weights[0], wires[0])
    # apply_ry(weights[1], wires[1])
    quit_rotation(weights[0], wires[0])
    apply_cnot(wires)


weights = np.random.random(size=[3, 2])

def circuit(image, weights):
    global state
    state = np.array(image, dtype=np.complex128)
    for i in range(len(weights)):
        block(weights[i], range(4))
    return state

def costfunc(params):
    cost = 0
    for i in range(len(BAS)):
        image = np.array(BAS[i])
        image = image / np.sqrt(np.sum(image ** 2))  # Normalize image
        output = circuit(image, params)
        if i < len(BAS) / 2:
            cost += output[3]
        else:
            cost -= output[3]
    return cost

params = np.random.random(size=[3, 2])
params_tensor = torch.tensor(params, requires_grad=True)
optimizer = torch.optim.SGD([params_tensor], lr=0.1)

if __name__ == '__main__':
    for k in range(100):
        if k % 20 == 0:
            print(f"Step {k}, cost: {costfunc(params)}")
        optimizer.zero_grad()
        loss = costfunc(params)
        loss.backward()
        optimizer.step()

    for image in BAS:
        fig = plt.figure(figsize=[1.8, 1.8])
        plt.imshow(np.reshape(image, [2, 2]), cmap="gray")
        plt.title(
            f"Exp. Val. = {circuit(image, params):.0f};"
            + f" Label = {'Bars' if circuit(image, params)[3] > 0 else 'Stripes'}",
            fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])

    plt.show()


