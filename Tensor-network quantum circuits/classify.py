# -*- coding: utf-8 -*-
# time: 2023/5/19 16:12
# file: classify.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np


BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
# j = 1

#show the image

# plt.figure(figsize=[3, 3])
# for i in BAS:
#     plt.subplot(2, 2, j)
#     j += 1
    # plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
    # plt.xticks([])
    # plt.yticks([])

def block(weights, wires):
    qml.RY(weights[0], wires=wires[0])
    qml.RY(weights[1], wires=wires[1])
    qml.CNOT(wires=wires)

dev = qml.device("default.qubit", wires=4)

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


weights = np.random.random(size=[3, 2])
fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(BAS[0], weights)
fig.set_size_inches((6, 3.5))

def costfunc(params):  #定义cost函数
    cost = 0
    for i in range(len(BAS)):
        if i < len(BAS) / 2:
            cost += circuit(BAS[i], params) #标签不符，增大cost
        else:
            cost -= circuit(BAS[i], params) #标签相符，减小cost
    return cost



if __name__ == '__main__':

    #定义参数大小和优化器

    params = np.random.random(size=[3, 2], requires_grad=True)
    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

    #训练
    for k in range(60):

        params = optimizer.step(costfunc, params)
        print(f"Step {k}, cost: {costfunc(params)}")

    print(f"Final parameters: {params}")


    #根据给出的图进行推理

    for image in BAS:
        qml.draw_mpl(circuit, expansion_strategy="device")(image, params)

        plt.figure(figsize=[1.8, 1.8])

        plt.imshow(np.reshape(image, [2, 2]), cmap="gray")

        plt.title(
            f"Exp. Val. = {circuit(image,params):.0f};"
            + f" Label = {'Bars' if circuit(image,params)>0 else 'Stripes'}",
            fontsize=8,
        )

        plt.xticks([])
        plt.yticks([])

    # plt.show()

