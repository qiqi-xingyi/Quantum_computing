import pennylane as qml
from pennylane import numpy as np


def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


dev = qml.device("default.qubit", wires=4)

#MPS

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


np.random.seed(1)
weights = np.random.random(size=[3, 2])
qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(weights)
fig.set_size_inches((6, 3))

def deep_block(weights, wires):
    qml.StronglyEntanglingLayers(weights, wires)

    dev = qml.device("default.qubit", wires=4)

shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
template_weights = [np.random.random(size=shape)] * 3
fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(template_weights)


#TTN
def block(weights, wires):
    qml.RX(weights[0], wires=wires[0])
    qml.RX(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)


dev = qml.device("default.qubit", wires=8)


@qml.qnode(dev, interface="autograd")
def circuit(template_weights):
    qml.TTN(
        wires=range(8),
        n_block_wires=2,
        block=block,
        n_params_block=2,
        template_weights=template_weights,
    )
    return qml.expval(qml.PauliZ(wires=7))


weights = np.random.random(size=[7, 2])
fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(weights)
fig.set_size_inches((4, 4))
