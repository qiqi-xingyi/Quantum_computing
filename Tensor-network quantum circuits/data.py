# -*- coding: utf-8 -*-
# time: 2023/5/25 22:32
# file: data.py
# author: Felix_Zhang
# email: yuqizhang247@gmail.com


import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    BAS = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    j = 1
    plt.figure(figsize=[3, 3])
    for i in BAS:
        plt.subplot(2, 2, j)
        j += 1
        plt.imshow(np.reshape(i, [2, 2]), cmap="gray")
        plt.xticks([])
        plt.yticks([])

    plt.show()