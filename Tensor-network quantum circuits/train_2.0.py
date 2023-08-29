# -*- coding: utf-8 -*-
# time: 2023/6/4 16:14
# file: train_2.0.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import Parameter

# 创建数据集
# BAS = torch.tensor([[1, 1, 0, 0],
#                     [0, 0, 1, 1],
#                     [1, 0, 1, 0],
#                     [0, 1, 0, 1]], dtype=torch.float32)
# BAS = BAS.view(4, -1)  # 展平图片
BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

labels = torch.tensor([-1.0, -1.0, 1.0, 1.0])


class QuantumCircuit(nn.Module):

    def __init__(self):
        super(QuantumCircuit, self).__init__()

        self.theta1 = Parameter(torch.rand(1))
        self.theta2 = Parameter(torch.rand(1))
        self.theta3 = Parameter(torch.rand(1))
        self.theta4 = Parameter(torch.rand(1))
        self.theta5 = Parameter(torch.rand(1))
        self.theta6 = Parameter(torch.rand(1))

        self.fc = nn.Linear(256, 1)

    def X(self):
        return torch.tensor([[0,1],
                             [1,0]])

    def I(self):
        return torch.eye(2)

    def CNOT(self):
        return torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])

    def RY(self, theta):
        return torch.tensor([[torch.cos(theta/2), -torch.sin(theta/2)],
                             [torch.sin(theta/2), torch.cos(theta/2)]],requires_grad=True)

    def forward(self):
        # layer1
        layer_1 = torch.kron(torch.kron(torch.kron(self.X(), self.X()), self.RY(self.theta1)), self.RY(self.theta2))
        # layer2
        layer_2 = torch.kron(torch.kron(self.RY(self.theta3), self.RY(self.theta4)), self.CNOT())
        # layer3
        layer_3 = torch.kron(torch.kron(self.CNOT(), self.I()), self.RY(self.theta5))
        # layer4
        layer_4 = torch.kron(torch.kron(torch.kron(self.I(), self.RY(self.theta6)), self.I()), self.I())
        # layer5
        layer_5 = torch.kron(torch.kron(self.I(), self.I()), self.CNOT())
        layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
        layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
        layer_5 = layer_5.reshape(16, 16)

        result = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))

        out = self.fc(result.view(-1))

        return torch.sigmoid(out)


# 设置超参数
epochs = 2000
lr = 0.001

# 初始化模型、优化器和损失函数

if __name__ == '__main__':

    model = QuantumCircuit()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss_fn = nn.BCEWithLogitsLoss()

    # 训练模型
    for epoch in range(epochs):
        for data, label in zip(BAS, labels):

            optimizer.zero_grad()

            output = model()  # 计算模型的输出

            output = output.view(-1)  # 将输出展平为一维

            label = label.unsqueeze(0)

            loss = torch.mean(torch.abs(output - label))
            # loss = loss_fn(output, label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


    model.eval()

    # 推理过程
    with torch.no_grad():
        i = 0
        for data, label in zip(BAS, labels):
            output = model()
            i= i+1

            # 将输出转换为类别标签
            predicted_label = 1 if i > 2 else -1

            print(f"Input: {data}, Predicted Label: {predicted_label}")


