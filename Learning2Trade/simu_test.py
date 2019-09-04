# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:37:57 2019
@author: Luofeng
"""
import pandas as pd
import numpy as np

from ou_simu import *
from run import *
from util import *

PARAMETER = {
    "learning_rate": 0.5,
    "criteria": "sharpe",
    "training_set": 0.8,
    "commission": 0.005,
    "position_size": 1,
    "preprocess": "diff",
    "time_interval": 15,
    "eta": 0.5
}


class Recorder:
    def __init__(self):
        self.times = 0
        self.rec = []

    def record(self, rec_list):
        self.times += 1
        self.rec.append(rec_list)

    def out(self):
        pd.DataFrame(self.rec).to_csv("out_rec.csv")


if __name__ == "__main__":
    # parameters definition
    TRAIN_PARA = {
        "length": 10000,
        "theta": 10,
        "mu": 0.0,
        "sigma": 10
        }

    time_interval = 20
    dd = False  # Objective: Sharpe ratio
    # dd = True    # Objective: return/semi-deviation
    rho = 0.025
    mu = 1
    comission = 0.005

    w = np.zeros(time_interval + 1)

    rec = Recorder()

    for k in range(1000):
        print("Loop: {}".format(k))
        # simulate OU process as log-price
        _data = ou_process(TRAIN_PARA["length"], TRAIN_PARA["theta"], TRAIN_PARA["mu"], TRAIN_PARA["sigma"]) * 100
        _data = ou_process(10000, 10, 0, 10)

        # training and testing
        return_test = direct_reinforce(_data, PARAMETER)

        rec.record([return_test])
    rec.out()