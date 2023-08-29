# -*- coding: utf-8 -*-
# time: 2023/5/29 16:34
# file: Tensor_net.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import numpy as np
import torch
import matplotlib.pyplot as plt

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


def quantum_circuit(theta):

    #layer1
    layer_1 = torch.kron(torch.kron(torch.kron(X(),X()),RY(theta)),RY(theta))
    print("layer_1:" , layer_1)

    #layer2
    layer_2 = torch.kron(torch.kron(RY(theta),RY(theta)),CNOT())
    print("layer_2:", layer_2)

    # layer3
    layer_3 =torch.kron(torch.kron(CNOT(),I()),RY(theta))
    print("layer_3:", layer_3)

    #layer4
    layer_4 = torch.kron(torch.kron(torch.kron(I(),RY(theta)),I()),I())
    print("layer_4:", layer_4)

    #layer5
    layer_5 = torch.kron(torch.kron(I(), I()), CNOT())
    print("layer_5:", layer_5)



if __name__ == '__main__':

    qubit_state_1 = torch.tensor([1, 0])
    qubit_state_2 = torch.tensor([1, 0])
    qubit_state_3 = torch.tensor([1, 0])
    qubit_state_4 = torch.tensor([1, 0])

    quantum_circuit(0.5)


