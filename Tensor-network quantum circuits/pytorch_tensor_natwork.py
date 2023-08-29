# -*- coding: utf-8 -*-
# time: 2023/5/25 23:31
# file: pytorch_tensor_natwork.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义图像数据和标签
BAS = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
labels = np.array([0, 0, 1, 1], dtype=np.int64)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #3个线性变换层
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        #前馈函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#创建模型实例和损失函数
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 定义训练函数
def train(model, inputs, labels, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #每次训练后打印loss值
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    print("Training completed!")

if __name__ == '__main__':

    # 转换为PyTorch张量
    inputs = torch.from_numpy(BAS)
    labels = torch.from_numpy(labels)

    # 训练模型
    train(model, inputs, labels, optimizer, criterion, num_epochs=1000)

    # 进行推理并显示结果
    with torch.no_grad():
        predicted_labels = model(inputs).argmax(dim=1)

    for i, image in enumerate(BAS):
        plt.figure(figsize=[1.8, 1.8])
        plt.imshow(np.reshape(image, [2, 2]), cmap="gray")
        plt.title(f"Label = {'Bars' if predicted_labels[i] == 0 else 'Stripes'}", fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.show()
