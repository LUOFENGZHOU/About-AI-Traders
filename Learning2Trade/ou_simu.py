# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 21:18:14 2019
@author: Luofeng
"""

import numpy as np
import matplotlib.pyplot as plt


def ou_process(n, theta, mu, sigma):
    data_list = np.zeros(n)
    for i in range(1, n):
        data_list[i] = data_list[i - 1] + theta * (mu - data_list[i - 1]) / n + sigma * np.random.normal(0, 1 / n)
    #    data_list = np.exp(data_list)
    plt.plot(data_list)
    plt.show()
    return data_list


if __name__ == "__main__":
    print("Simulating OU Process...")
    ou_process(10000, 10, 0, 10)
