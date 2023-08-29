# -*- coding: utf-8 -*-
# time: 2023/8/17 22:02
# file: brick_wall_network2.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#Identity matrix
def I():
    return torch.eye(2, requires_grad=False)

def PauliZ():
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, requires_grad=True)

def CNOT():
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], requires_grad=False)

def partial_trace(rho, keep, dims, optimize=False):

    keep = np.asarray(keep)  # 将 `keep` 数组转换为 NumPy 数组，方便后续处理。
    dims = np.asarray(dims)  # 将 `dims` 数组转换为 NumPy 数组，方便后续处理。
    Ndim = dims.size  # 复合量子系统的子系统数目。
    Nkeep = np.prod(dims[keep])  # 计算偏迹后保留子系统的维度。
    # print("keep:" , Nkeep)

    # 创建 einsum 缩并的索引列表。
    idx1 = [i for i in range(Ndim)]  # 缩并的第一部分的索引列表。
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]  # 缩并的第二部分的索引列表。
    # 将输入的密度矩阵进行形状变换，为 einsum 缩并做准备。
    rho_a = rho.reshape(np.tile(dims, 2))
    # 使用 einsum 缩并计算偏迹。
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=False)
    # 将结果矩阵重新调整为迹掉子系统后的期望形状。
    rho_a = rho_a.reshape(Nkeep, Nkeep)

    return rho_a


def u(phi, theta, omega):

    U11 = torch.cos(theta / 2) * torch.exp(-1j * (phi + omega) / 2)
    U12 = -torch.sin(theta / 2) * torch.exp(1j * (phi - omega) / 2)
    U21 = torch.sin(theta / 2) * torch.exp(-1j * (phi - omega) / 2)
    U22 = torch.cos(theta / 2) * torch.exp(1j * (phi + omega) / 2)

    U11 = U11.unsqueeze(1)
    U12 = U12.unsqueeze(1)
    U21 = U21.unsqueeze(1)
    U22 = U22.unsqueeze(1)

    U = torch.cat((U11, U12, U21, U22), dim=1)
    U = U.reshape(2, 2)

    return U


def compute_entropy(input_rho):

    rho_tensor = torch.tensor(input_rho)
    #计算纠缠熵
    evals = torch.linalg.eigvalsh(rho_tensor)
    entropy = -torch.sum(evals * torch.log2(evals))
    return  entropy


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def convert_to_bloch_vector(rho):

    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real

    return [ax, ay, az]

def plot_bloch_sphere(bloch_vectors):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(30, 45)
    ax.dist = 7

    x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
    u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

    ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
    ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
    ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
    ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
    ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
    ax.text(0,-1.9, 0, r"|i–⟩", color="black", fontsize=16)

    ax.scatter(
        bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
    )


class Brick_Wall_Network(nn.Module):

    def __init__(self):
        super(Brick_Wall_Network, self).__init__()

        self.phi_1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta_1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.omega_1 = nn.Parameter(torch.rand(1), requires_grad=True)

        self.phi_2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta_2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.omega_2 = nn.Parameter(torch.rand(1), requires_grad=True)

        self.phi_3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta_3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.omega_3 = nn.Parameter(torch.rand(1), requires_grad=True)

        self.phi_4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.theta_4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.omega_4 = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, input_state):

        input_state = torch.complex(input_state, torch.zeros_like(input_state))

        layer_1 = torch.kron(torch.kron(torch.kron(u(self.phi_1 , self.theta_1 , self.omega_1) , u(self.phi_2 , self.theta_3 , self.omega_4)) ,
                             u(self.phi_3 , self.theta_3 , self.omega_3)) , u(self.phi_4 , self.theta_4 , self.omega_4))

        layer_2 = torch.kron(torch.kron(CNOT() , I()) , I())
        layer_3 = torch.kron(torch.kron(I() , CNOT()) , I())
        layer_4 = torch.kron(torch.kron(I(), I()), CNOT())

        layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
        layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
        layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
        layer_5 = layer_5.reshape(16, 16)

        layer_2 = torch.complex(layer_2 , torch.zeros_like(layer_2))
        layer_3 = torch.complex(layer_3, torch.zeros_like(layer_3))
        layer_4 = torch.complex(layer_4, torch.zeros_like(layer_4))
        layer_5 = torch.complex(layer_5, torch.zeros_like(layer_5))

        U = torch.matmul(layer_5,torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))

        M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())

        alpha = torch.matmul(U, input_state)

        conjugate_alpha = torch.conj(alpha.float())
        transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
        output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())

        return output_value

brick_wall_network = Brick_Wall_Network()

if __name__ == '__main__':

    brick_wall_network.train()

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
    learning_rate = 0.002
    num_epochs = 100

    # 定义优化器
    optimizer = optim.SGD(brick_wall_network.parameters(), lr=learning_rate)

    loss_fn = nn.MSELoss()

    # 迭代训练
    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_state, label_tensor in zip(states, label_tensors):
            # 将梯度归零
            optimizer.zero_grad()
            # 前向传播
            output = brick_wall_network(input_state)
            # print("output:", output)
            label = label_tensor.view(output.shape)
            # 计算损失值
            # loss = torch.mean(torch.abs(output - label))
            loss = loss_fn(output, label)
            loss.backward()

            # 更新参数
            optimizer.step()
            # 累计损失值
            running_loss += loss.item()

        # 打印平均损失值
        average_loss = running_loss / len(states)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

        #############################################################################################

    # 设置模型为推理模式
    brick_wall_network.eval()


    #固定参数，用partial trace分解
    print('####################################################################')
    # 获取训练后的参数

    model = Brick_Wall_Network()

    # 固定参数
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    layer_1 = torch.kron(
        torch.kron(torch.kron(u(model.phi_1, model.theta_1, model.omega_1), u(model.phi_2, model.theta_3, model.omega_4)),
                   u(model.phi_3, model.theta_3, model.omega_3)), u(model.phi_4, model.theta_4, model.omega_4))

    layer_2 = torch.kron(torch.kron(CNOT(), I()), I())
    layer_3 = torch.kron(torch.kron(I(), CNOT()), I())
    layer_4 = torch.kron(torch.kron(I(), I()), CNOT())

    layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
    layer_5 = layer_5.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    layer_5 = torch.transpose(torch.transpose(layer_5, 1, 2), 5, 6)
    layer_5 = layer_5.reshape(16, 16)

    layer_2 = torch.complex(layer_2, torch.zeros_like(layer_2))
    layer_3 = torch.complex(layer_3, torch.zeros_like(layer_3))
    layer_4 = torch.complex(layer_4, torch.zeros_like(layer_4))
    layer_5 = torch.complex(layer_5, torch.zeros_like(layer_5))

    U = torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1))))

    state_for_e = states[0]

    state_for_e = torch.complex(state_for_e, torch.zeros_like(state_for_e))
    alpha = torch.matmul(U, state_for_e)

    # print(alpha.size())

    alpha_conj = torch.conj(alpha)  # 复共轭
    alpha_conj_t = torch.transpose(alpha_conj, 0, 1)  # 转置

    rho = torch.kron(alpha, alpha_conj_t)
    # print(rho.size())

    evals = torch.linalg.eigvalsh(rho)
    # print(evals)
    # for eval in evals:

    entropy = -torch.sum(evals * torch.log2(evals))
    # print("the entropy is:" , entropy)

    # 密度函数 rho
    rho_array = rho.numpy()

    qubit_1_rho = partial_trace(rho_array, [0], [2, 2, 2, 2])

    qubit_2_rho = partial_trace(rho_array, [1], [2, 2, 2, 2])

    qubit_3_rho = partial_trace(rho_array, [2], [2, 2, 2, 2])

    qubit_4_rho = partial_trace(rho_array, [3], [2, 2, 2, 2])



    ###############################################################


    print("the matrix of qubit_1:", qubit_1_rho)
    print("the matrix of qubit_2:", qubit_2_rho)
    print("the matrix of qubit_3:", qubit_3_rho)
    print("the matrix of qubit_4:", qubit_4_rho)

    not_haar_samples = [qubit_1_rho, qubit_2_rho , qubit_3_rho , qubit_4_rho]

    not_haar_bloch_vectors = np.array([convert_to_bloch_vector(qubit_1_rho) , convert_to_bloch_vector(qubit_2_rho),
                                       convert_to_bloch_vector(qubit_3_rho),convert_to_bloch_vector(qubit_4_rho)])

    print("not_haar_bloch_vectors :" , not_haar_bloch_vectors)

    plot_bloch_sphere(not_haar_bloch_vectors)

    plt.show()





    


