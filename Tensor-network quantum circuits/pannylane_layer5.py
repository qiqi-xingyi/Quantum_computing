# -*- coding: utf-8 -*-
# time: 2023/5/29 21:59
# file: pannylane_layer5.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com


import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def my_circuit():
    qml.CNOT(wires=[1, 3])
    return qml.state()

result = my_circuit()

if __name__ == '__main__':
        print(result)
