# -*- coding: utf-8 -*-
# time: 2023/6/2 23:09
# file: grad.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.scalar_param = nn.Parameter(torch.tensor(0.5), requires_grad=True)  # 创建一个只有一维的参数

    def forward(self, x):
        output = x * self.scalar_param
        return output

model = MyModel()

if __name__ == '__main__':

    # 计算梯度
    input = torch.tensor(2.0)
    output = model(input)
    output.backward()

    # 获取参数梯度
    param_gradient = model.scalar_param.grad
    print(param_gradient)
