# -*- coding: utf-8 -*-
# time: 2023/5/29 21:48
# file: Panny_TTN.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import pennylane as qml
from pennylane import numpy as np

def block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)

dev = qml.device("default.qubit", wires=4)

BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

@qml.qnode(dev, interface="autograd")
def circuit(image, template_weights):
    qml.BasisStatePreparation(image, wires=range(4))
    qml.TTN(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )

    return qml.expval(qml.PauliZ(wires=3))


if __name__ == '__main__':
    weights = np.random.random(size=[3, 2])
    fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(BAS[0], weights)
    fig.set_size_inches((6, 3.5))

