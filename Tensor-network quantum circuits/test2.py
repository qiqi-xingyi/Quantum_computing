# -*- coding: utf-8 -*-
# time: 2023/6/5 13:24
# file: test2.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim


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

        output_value = self.theta1*self.theta2*self.theta3*self.theta4*self.theta5*self.theta6

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


