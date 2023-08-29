#Qubit_ratation

import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device("default.qubit", wires=1)

@qml.qnode(dev1, interface="autograd")
def circuit(params):

    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

def cost(x):
    return circuit(x)

init_params = np.array([1.8, 1.62], requires_grad=True)

# initialise the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 500
# set the initial parameter values
params = init_params

if __name__ == '__main__':


    for i in range(steps):
        # update the circuit parameters
        params = opt.step(cost, params)

        if (i + 1) % 5 == 0:
            print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

    print("Optimized rotation angles: {}".format(params))





