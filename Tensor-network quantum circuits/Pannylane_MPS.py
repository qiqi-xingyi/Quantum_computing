# -*- coding: utf-8 -*-
# time: 2023/5/25 22:12
# file: Pannylane_MPS.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

#用pannylane构建MPS量子线路
import pennylane as qml
from pennylane import numpy as np


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)

dev = qml.device("default.qubit", wires=4)


@qml.qnode(dev, interface="autograd")
def circuit(template_weights):
    qml.MPS(
        wires=range(4),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=3))

if __name__ == '__main__':


    np.random.seed(1)
    weights = np.random.random(size=[3, 2])
    qml.drawer.use_style("black_white")
    fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(weights)
    fig.set_size_inches((6, 3))

