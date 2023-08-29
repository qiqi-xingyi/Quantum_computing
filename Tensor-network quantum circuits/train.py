# -*- coding: utf-8 -*-
# time: 2023/5/30 21:13
# file: train.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

#加载数据集
BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
# j = 1
# show the image
# plt.figure(figsize=[3, 3])
# for i in BAS:
#     plt.subplot(2, 2, j)
#     j += 1
#     plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
#     plt.xticks([])
#     plt.yticks([])
#
# plt.show()

# 定义量子门操作函数
def rx(theta):
    return torch.tensor([[torch.cos(theta/2), -1j*torch.sin(theta/2)],
                         [-1j*torch.sin(theta/2), torch.cos(theta/2)]]).requires_grad_()

def RY(theta):
    return torch.tensor([[torch.cos(theta/2), -torch.sin(theta/2)],
                         [torch.sin(theta/2), torch.cos(theta/2)]]).requires_grad_()

def CNOT():

    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]])

def X():

    return torch.tensor([[0,1],
                         [1,0]])

def I():
    return torch.eye(2)

def PauliZ():
    return torch.tensor([[1 , 0],
                         [0 ,-1]])

def measure(alpha):

    M = torch.kron(torch.kron(torch.kron(I(),I()),I()),PauliZ())

    # print("the size of M : ", M.size())

    conjugate_alpha =  torch.conj(alpha)

    transpose_alpha = torch.transpose(conjugate_alpha , 0 , 1 )

    # print("the size of alpha* : " , transpose_alpha.size())

    maesure_result = torch.matmul(torch.matmul(transpose_alpha.float() , M.float()) ,alpha)

    return maesure_result


class QuantumCircuit(nn.Module):

    def __init__(self):
        super(QuantumCircuit, self).__init__()

        #定义可训练参数
        self.theta1 = nn.Parameter(torch.rand(1))
        self.theta2 = nn.Parameter(torch.rand(1))
        self.theta3 = nn.Parameter(torch.rand(1))
        self.theta4 = nn.Parameter(torch.rand(1))
        self.theta5 = nn.Parameter(torch.rand(1))
        self.theta6 = nn.Parameter(torch.rand(1))

    def forward(self , input_state):

        #定义量子网络
        layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(self.theta1)), RY(self.theta2))
        layer_2 = torch.kron(torch.kron(RY(self.theta3), RY(self.theta4)), CNOT())

        layer_3 = torch.kron(torch.kron(CNOT(), I()), RY(self.theta5))
        layer_4 = torch.kron(torch.kron(torch.kron(I(), RY(self.theta6)), I()), I())
        layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
        layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
        layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
        layer_5 = layer_5.reshape(16, 16)

        U = torch.matmul(layer_5,torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))
        alpha = torch.matmul(U, input_state.float())
        output_value = measure(alpha.float())

        return output_value


if __name__ == '__main__':

    quantum_circuit = QuantumCircuit()

    # 把图片编码成量子态
    BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    input_states = []
    for image in BAS:
        quantum_state = torch.zeros(4, 2, 1)
        for i, pixel in enumerate(image):
            quantum_state[i][int(pixel)][0] = 1
        input_states.append(quantum_state)

    #构建输入态
    states =  []

    states.append(torch.kron(torch.kron(torch.kron(input_states[0][0] , input_states[0][1]),input_states[0][2]),
                           input_states[0][3]))
    states.append(torch.kron(torch.kron(torch.kron(input_states[1][0], input_states[1][1]), input_states[1][2]),
                           input_states[1][3]))
    states.append(torch.kron(torch.kron(torch.kron(input_states[2][0], input_states[2][1]), input_states[2][2]),
                           input_states[2][3]))
    states.append(torch.kron(torch.kron(torch.kron(input_states[3][0], input_states[3][1]), input_states[3][2]),
                           input_states[3][3]))

    #数据集标签
    label = [-1.0, -1.0, 1.0, 1.0]
    label_tensors = [torch.tensor(element, dtype=torch.float32) for element in label]

    #二分类训练器

    # 设置超参数
    learning_rate = 0.01
    num_epochs = 100

    # 定义损失函数
    criterion = nn.BCELoss()

    # 定义优化器
    optimizer = optim.SGD(quantum_circuit.parameters(), lr=learning_rate)

    # 迭代训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_state, label_tensor in zip(states, label_tensors):

            # 将梯度归零
            optimizer.zero_grad()

            # 前向传播
            output = quantum_circuit(input_state)
            print("output:" , output)

            label = label_tensor.view(output.shape)
            # 计算损失值
            # loss = criterion(output, label)
            loss = torch.mean(torch.abs(output - label))

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 累计损失值
            running_loss += loss.item()

        # 打印平均损失值
        average_loss = running_loss / len(states)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

