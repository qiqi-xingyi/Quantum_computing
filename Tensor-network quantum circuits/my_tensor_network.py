# -*- coding: utf-8 -*-
# time: 2023/5/18 20:45
# file: my_tensor_network.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

# j = 1
# plt.figure(figsize=[3, 3])
# for i in BAS:
#     plt.subplot(2, 2, j)
#     j += 1
#     plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
#     plt.xticks([])
#     plt.yticks([])


class RY(nn.Module):
    def __init__(self, theta):
        super(RY, self).__init__()
        self.theta = nn.Parameter(theta)

    def forward(self, x):
        cos = torch.cos(self.theta / 2)
        sin = torch.sin(self.theta / 2)
        matrix = torch.stack([cos, -sin, sin, cos]).reshape(2, 2)
        return torch.matmul(matrix, x)


class CNOT(nn.Module):
    def __init__(self):
        super(CNOT, self).__init__()

    def forward(self, x):
        matrix = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float32)
        return torch.matmul(matrix, x)


class QuantumCircuit(nn.Module):
    def __init__(self):
        super(QuantumCircuit, self).__init__()
        self.block_weights = nn.Parameter(torch.randn(3, 2))

    def block(self, inputs):
        outputs = torch.zeros_like(inputs)
        outputs[:, 0] = torch.cos(self.block_weights[:, 0]) * inputs[:, 0] - torch.sin(self.block_weights[:, 0]) * inputs[:, 1]
        outputs[:, 1] = torch.cos(self.block_weights[:, 1]) * inputs[:, 1] + torch.sin(self.block_weights[:, 1]) * inputs[:, 0]
        return outputs

    def forward(self, inputs):
        x = inputs
        x = self.block(x)
        x = torch.reshape(x, [2, 2])
        return x


model = QuantumCircuit()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def costfunc(params):
    cost = 0
    for i in range(len(BAS)):
        inputs = torch.tensor(BAS[i], dtype=torch.float32)
        if i < len(BAS) / 2:
            output = model(inputs)
            cost += output[1, 1] - output[1, 0] - output[0, 1] + output[0, 0]
        else:
            output = model(inputs)
            cost -= output[1, 1] - output[1, 0] - output[0, 1] + output[0, 0]
    return cost


if __name__ == '__main__':

    #train
    for k in range(100):
        optimizer.zero_grad()
        loss = costfunc(model.parameters())
        loss.backward()
        optimizer.step()

        if k % 20 == 0:
            print(f"Step {k}, cost: {loss.item()}")

    #infer
    for image in BAS:
        inputs = torch.tensor(image, dtype=torch.float32)
        output = model(inputs)

        plt.figure(figsize=[1.8, 1.8])
        plt.imshow(output.detach().numpy(), cmap="gray")
        plt.title(f"Exp. Val. = {output[1, 1] - output[1, 0] - output[0, 1] + output[0, 0]:.0f};"
                  f" Label = {'Bars' if output[1, 1] - output[1, 0] - output[0, 1] + output[0, 0] > 0 else 'Stripes'}",
                  fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.show()

