# -*- coding: utf-8 -*-
# time: 2023/6/5 13:24
# file: test2.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim


# 定义量子门操作函数
def RY(theta):
    U11 = torch.cos(theta / 2)
    U12 = -torch.sin(theta / 2)
    U21 = torch.sin(theta / 2)
    U22 = torch.cos(theta / 2)
    U11 = U11.unsqueeze(1)
    U22 = U22.unsqueeze(1)
    U12 = U12.unsqueeze(1)
    U21 = U21.unsqueeze(1)
    U = torch.cat( (U11,U12,U21,U22) , dim= 1)
    U = U.reshape(2, 2)
    return U

def CNOT():
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], requires_grad=False)

def X():
    return torch.tensor([[0, 1],
                         [1, 0]], requires_grad=False)

def I():
    return torch.eye(2, requires_grad=False)

def PauliZ():
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, requires_grad=True)


class QuantumCircuit(nn.Module):
    def __init__(self):
        super(QuantumCircuit, self).__init__()
        # 定义可训练参数
        self.theta1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta5 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta6 = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, input_state):
        # 定义量子网络
        layer_1 = torch.kron(torch.kron(torch.kron(X(), X()), RY(self.theta1)), RY(self.theta2))
        layer_2 = torch.kron(torch.kron(RY(self.theta3), RY(self.theta4)), CNOT())

        U = torch.matmul(layer_2, layer_1)

        alpha = torch.matmul(U, input_state.float())
        M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())
        conjugate_alpha = torch.conj(alpha.float())
        transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
        output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())


        return output_value

quantum_circuit = QuantumCircuit()

if __name__ == '__main__':

    quantum_circuit.train()

    # 把图片编码成量子态
    BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    input_states = []
    for image in BAS:
        quantum_state = torch.zeros(4, 2, 1)
        for i, pixel in enumerate(image):
            quantum_state[i][int(pixel)][0] = 1
        input_states.append(quantum_state)

    # 构建输入态
    states = []
    for input_state in input_states:
        state = torch.kron(input_state[0], torch.kron(input_state[1], torch.kron(input_state[2], input_state[3])))
        states.append(state)


    # 数据集标签
    label = [-1.0, -1.0, 1.0, 1.0]
    label_tensors = [torch.tensor(element, dtype=torch.float32) for element in label]

    # 设置超参数
    learning_rate = 0.2
    num_epochs = 100

    # 定义优化器
    optimizer = optim.SGD(quantum_circuit.parameters(), lr=learning_rate)

    loss_fn = nn.MSELoss()

    # 迭代训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_state, label_tensor in zip(states, label_tensors):
            # 将梯度归零
            optimizer.zero_grad()
            # 前向传播
            output = quantum_circuit(input_state)
            # print("output:", output)
            label = label_tensor.view(output.shape)
            # 计算损失值
            # loss = torch.mean(torch.abs(output - label))
            loss = loss_fn(output, label)
            loss.backward()
            #
            for name, param in quantum_circuit.named_parameters():
                # if param.grad is not None:
                print("the grad:", name, param.grad)

            print("*********************************")

            # 更新参数
            optimizer.step()
            # 累计损失值
            running_loss += loss.item()


        # 打印平均损失值
        average_loss = running_loss / len(states)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")


