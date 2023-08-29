# -*- coding: utf-8 -*-
# time: 2023/5/29 16:34
# file: Tensor_net.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np
import torch
import matplotlib.pyplot as plt


#加载数据集
BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
j = 1
# show the image
plt.figure(figsize=[3, 3])
for i in BAS:
    plt.subplot(2, 2, j)
    j += 1
    plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
    plt.xticks([])
    plt.yticks([])

plt.show()

# 定义量子门操作函数
def rx(theta):

    return torch.tensor([[np.cos(theta/2), -1j*np.sin(theta/2)],
                        [-1j*np.sin(theta/2), np.cos(theta/2)]])

def RY(theta):

    return torch.tensor([[np.cos(theta/2), -np.sin(theta/2)],
                        [np.sin(theta/2), np.cos(theta/2)]])

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

def quantum_circuit():

    theta1 = torch.rand(1)
    theta2 = torch.rand(1)
    theta3 = torch.rand(1)
    theta4 = torch.rand(1)
    theta5 = torch.rand(1)
    theta6 = torch.rand(1)

    #layer1
    layer_1 = torch.kron(torch.kron(torch.kron(X(),X()),RY(theta1)),RY(theta2))
    # print("layer_1:" , layer_1)

    #layer2
    layer_2 = torch.kron(torch.kron(RY(theta3),RY(theta4)),CNOT())
    # print("layer_2:", layer_2)

    # layer3
    layer_3 =torch.kron(torch.kron(CNOT(),I()),RY(theta5))
    # print("layer_3:", layer_3)

    #layer4
    layer_4 = torch.kron(torch.kron(torch.kron(I(),RY(theta6)),I()),I())
    # print("layer_4:", layer_4)

    #layer5
    layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
    layer_5 = layer_5.reshape(2,2,2,2,2,2,2,2)
    # print("55" , layer_5)
    layer_5 = torch.transpose(torch.transpose( layer_5, 1 , 2 ), 5, 6)
    layer_5 = layer_5.reshape(16,16)
    # print("layer_5:", layer_5)

    return torch.matmul(layer_5,torch.matmul(layer_4,torch.matmul(layer_3,torch.matmul(layer_2,layer_1))))


def measure(alpha):

    M = torch.kron(torch.kron(torch.kron(I(),I()),I()),PauliZ())

    print("the size of M : ", M.size())

    conjugate_alpha =  torch.conj(alpha)

    transpose_alpha = torch.transpose(conjugate_alpha , 0 , 1 )

    print("the size of alpha* : " , transpose_alpha.size())

    maesure_result = torch.matmul(torch.matmul(transpose_alpha.float() , M.float()) ,alpha)

    return maesure_result


if __name__ == '__main__':

    qubit_state_1 = torch.tensor([[1],
                                 [0]])
    qubit_state_2 = torch.tensor([[1],
                                 [0]])
    qubit_state_3 = torch.tensor([[1],
                                 [0]])
    qubit_state_4 = torch.tensor([[1],
                                 [0]])

    state = torch.kron(torch.kron(torch.kron(qubit_state_1 ,qubit_state_2),qubit_state_3),qubit_state_4)
    print(state)
    print("the size of state:" , state.size())

    circuits = quantum_circuit()
    print("the size of U:", circuits.size())

    U = circuits.float()
    # print("U:" , U)

    alpha = torch.matmul(U , state.float())
    # print("alpha:" , alpha)
    print("the size of alpha:", alpha.size())
    # print("measure: " , measure(alpha))

    maesured_value = measure(alpha.float())
    print("the maesured value:" , maesured_value)




