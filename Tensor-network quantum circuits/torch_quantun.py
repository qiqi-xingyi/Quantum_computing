# -*- coding: utf-8 -*-
# time: 2023/6/3 18:54
# file: torch_quantun.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com


import torch
import torch.nn as nn
import torchquantum as tq
import torch.optim as optim


class QuantumCircuit(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.ry1 = tq.RY(has_params=True, trainable=True)
        self.ry2 = tq.RY(has_params=True, trainable=True)
        self.ry3 = tq.RY(has_params=True, trainable=True)
        self.ry4 = tq.RY(has_params=True, trainable=True)
        self.ry5 = tq.RY(has_params=True, trainable=True)
        self.ry6 = tq.RY(has_params=True, trainable=True)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True)

        qdev.h(wires=0)
        qdev.h(wires=1)
        self.ry1(qdev, wires=0)
        self.ry2(qdev, wires=1)
        qdev.cnot(wires=[0, 1])
        qdev.h(wires=2)
        qdev.h(wires=3)
        self.ry3(qdev, wires=2)
        self.ry4(qdev, wires=3)
        qdev.cnot(wires=[2, 3])
        self.ry5(qdev, wires=2)
        qdev.i(wires=1)
        self.ry6(qdev, wires=0)
        qdev.cnot(wires=[2, 3])

        return qdev.measure_all()  # measure all qubits and return expectation values


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
            output = quantum_circuit(input_state.unsqueeze(0))  # add batch dimension
            label = label_tensor.view(output.shape)
            # 计算损失值
            loss = loss_fn(output, label)
            loss.backward()

            for name, param in quantum_circuit.named_parameters():
                print("the grad:", name, param.grad)

            print("*********************************")

            # 更新参数
            optimizer.step()
            # 累计损失值
            running_loss += loss.item()

        # 打印平均损失值
        average_loss = running_loss / len(states)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")


