import pytorch as torch
import numpy as np
import cmath

# 定义旋转门的参数
params = torch.tensor([1.5, 1.6], dtype=torch.float64, requires_grad=True)
step_size = 0.1
num_iterations = 1000
tolerance = 1e-5

def rx_gate(theta):
    return torch.tensor([[torch.cos(theta / 2), -1j * torch.sin(theta / 2)],
                     [-1j * torch.sin(theta / 2), torch.cos(theta / 2)]])

def ry_gate(theta):
    return torch.tensor([[torch.cos(theta / 2), -torch.sin(theta / 2)],
                     [torch.sin(theta / 2), torch.cos(theta / 2)]])

def initial_state():
    return torch.tensor([1, 0], dtype=torch.complex128)

def measure(state):
    prob = torch.abs(state) ** 2
    if np.random.uniform(0, 1) < prob[0]:
        return 0
    else:
        return 1

def circuit(params):
    state = initial_state()
    state = torch.matmul(rx_gate(params[0]), state)
    state = torch.matmul(ry_gate(params[1]), state)
    return measure(state)

def cost(params):
    expectation = 0
    for i in range(100):
        expectation += circuit(params)
    expectation /= 100
    return expectation

if __name__ == '__main__':
    # 初始化参数
    prev_cost = cost(params)

    # 开始迭代
    for i in range(num_iterations):
        cost_value = cost(params)
        cost_value.backward()

        with torch.no_grad():
            params -= step_size * params.grad

            params.grad.zero_()

            curr_cost = cost(params)

            if torch.abs(curr_cost - prev_cost) < tolerance:
                print("Converged after %d iterations." % i)
                break

            prev_cost = curr_cost

    print("Optimized parameters: ", params)
    print("Optimized cost: ", curr_cost)
