# -*- coding: utf-8 -*-
# time: 2023/5/30 17:13
# file: quantum_c.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=4)
# 定义量子线路
@qml.qnode(dev)

def quantum_circuit(theta):
    qml.BasisState([0, 0, 0, 0], wires=[0, 1, 2, 3])

    qml.PauliX(wires=0)
    qml.PauliX(wires=1)
    qml.RY(theta , wires=2)
    qml.RY(theta , wires=3)

    qml.RY(theta, wires=0)
    qml.RY(theta, wires=1)
    qml.CNOT(wires=[2, 3])

    qml.CNOT(wires=[0, 1])
    qml.RY(theta, wires=3)

    qml.RY(theta, wires=1)
    qml.CNOT(wires=[1, 3])

    return qml.expval(qml.PauliZ(3))

if __name__ == '__main__':
    qml.draw(quantum_circuit)

    # 运行量子线路
    theta = 0.5

    result = quantum_circuit(theta)

    print("Expectation value:", result)



