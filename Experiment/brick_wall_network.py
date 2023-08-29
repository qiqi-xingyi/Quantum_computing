# -*- coding: utf-8 -*-
# time: 2023/8/10 16:27
# file: quantum_net.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com
# brick wall network

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#定义量子门

def g(x):

    # Define Pauli matrices
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)

    # Define 4x4 identity matrix
    identity = torch.eye(4, dtype=torch.complex64)

    # Define the SU(4) generators
    G1 = torch.kron(sigma_x, identity)
    G2 = torch.kron(sigma_y, identity)
    G3 = torch.kron(sigma_z, identity)
    G4 = torch.kron(identity, sigma_x)
    G5 = torch.kron(identity, sigma_y)
    G6 = torch.kron(identity, sigma_z)

    # Return the matrices
    if x == 1:
        return G1

    if x == 2:
        return G2

    if x == 3:
        return G3

    if x == 4:
        return G4

    if x == 5:
        return G5

    if x == 6:
        return G6

#Identity matrix
def I():
    return torch.eye(2, requires_grad=False)

def PauliZ():
    return torch.tensor([[1, 0], [0, -1]], dtype=torch.float32, requires_grad=True)


class Brick_Wall_Network(nn.Module):

    def __init__(self):
        super(Brick_Wall_Network, self).__init__()

        self.a_23_1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_23_2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_23_3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_23_4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_12_1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_12_2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_12_3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_12_4 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_34_1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_34_2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_34_3 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.a_34_4 = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, input_state):

        def u_12(self):
            u = self.a_12_1 * g(1) + self.a_12_2 * g(2) + self.a_12_3 * g(3) + self.a_12_4 * g(4)
            exp_u = torch.exp(u)

            return exp_u

        def u_23(self):
            u = self.a_23_1 * g(1) + self.a_23_2 * g(2) + self.a_23_3 * g(3) + self.a_23_4 * g(4)
            exp_u = torch.exp(u)

            return exp_u

        def u_34(self):
            u = self.a_34_1 * g(1) + self.a_34_2 * g(2) + self.a_34_3 * g(3) + self.a_34_4 * g(4)
            exp_u = torch.exp(u)

            return exp_u

        layer_1 = torch.kron(torch.kron(I() , u_23(self)) , I())
        layer_2 = torch.kron(u_12(self) , u_34(self))
        layer_3 = torch.kron(torch.kron(I() , u_23(self)) , I())
        layer_4 = torch.kron(u_12(self) , u_34(self))
        layer_5 = torch.kron(torch.kron(I() , u_23(self)) , I())
        layer_6 = torch.kron(u_12(self) , u_34(self))

        U = torch.matmul(layer_6, torch.matmul(layer_5, torch.matmul(layer_4, torch.matmul(layer_3, torch.matmul(layer_2, layer_1)))))

        M = torch.kron(torch.kron(torch.kron(I(), I()), I()), PauliZ())

        alpha = torch.matmul(U, input_state.float())

        conjugate_alpha = torch.conj(alpha.float())
        transpose_alpha = torch.transpose(conjugate_alpha, 0, 1)
        output_value = torch.matmul(torch.matmul(transpose_alpha.float(), M.float()), alpha.float())

        return output_value

Brick_Wall_Network = Brick_Wall_Network()
