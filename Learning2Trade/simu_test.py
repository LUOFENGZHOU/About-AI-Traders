# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 13:37:57 2019

@author: Luofeng
"""
import pandas as pd
import numpy as np

from ou_simu import *
from run import *

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
            "length" : 300,
            "theta" : 1,
            "mu" : 0.0,
            "sigma" : 0.3,
            "train_num" : 200,}
    
    time_interval = 20
    dd = False  # Objective: Sharpe ratio
    #dd = True    # Objective: return/semi-deviation
    rho = 0.025
    mu = 1
    comission = 0.005
    
    w = np.zeros(time_interval+1)

    rec = Recorder()
    
    for k in range(1000):
        print("Loop: {}".format(k))
        # simulate OU process as log-price
        _data = ou_process(TRAIN_PARA["length"],TRAIN_PARA["theta"],TRAIN_PARA["mu"],TRAIN_PARA["sigma"]) * 100
        train_data = _data[:TRAIN_PARA["train_num"]]
        test_data = _data[TRAIN_PARA["train_num"]:]
    
        # training and testing
        ws, trading_sr, trading_ddratio, BH_sr, BH_ddratio, capital_rtn, BH_rtn = DirectRL(time_interval, train_data, dd, rho, mu, comission,w) 
        w1, trading_sr1, trading_ddratio1, BH_sr1, BH_ddratio1, capital_rtn1, BH_rtn1 = DirectRL(time_interval, test_data, dd, rho, mu, comission,ws) 
        
        rec.record([str(dd), trading_sr, trading_ddratio, BH_sr, BH_ddratio, capital_rtn[-1], BH_rtn[-1], trading_sr1, trading_ddratio1, BH_sr1, BH_ddratio1, capital_rtn1[-1], BH_rtn1[-1]])
    rec.out()