import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from util import * 

from pandas_datareader import data
from hyperopt import fmin, tpe, hp, partial




PARAMETER = {
    "learning_rate": 0.5,
    "criteria": "sharpe", # choose between sharpe, donwside_ratio and downside_risk
    "training_set": 0.8,
    "commission": 0.005,
    "position_size": 1, # choose between zero and one
    "preprocess": "log_dif", # choose between log_dif, dif and others
    "time_interval": 10,
    "eta": 0.5
}


def RL_Trader(input, parameters, theta):
    if theta[0] == 0:
#        theta = np.zeros(parameters["time_interval"] + 2)  # a constant, decision and returns in the past
#        theta = np.random.rand(parameters["time_interval"] + 2)
        theta = np.zeros(parameters["time_interval"] + 2)
    hat_list = get_hat_list(input[:parameters["time_interval"]],
                            parameters)  # initialize moving average estimators, e.g. A and B in the paper.
    data_in = np.append([1, 1], input[:parameters["time_interval"]])
    prev_dF_dTheta = False
    prev_decision = 1
    return_list = []
    return_list.extend(input[:parameters["time_interval"]])
    for i in range(parameters["time_interval"] + 1, len(input)):  # main loop: when new info comes in

        theta, decision, dF_dTheta, R_t = update_theta(theta, data_in, input[i], parameters, hat_list, prev_dF_dTheta,
                                                       prev_decision)  # update the network

        # preparing for the next loop
        data_in = np.append(data_in[:-1], input[i])
        data_in[0] = 1
        data_in[1] = decision
        hat_list = get_hat_list(R_t, parameters, hat_list)  # update hat_list
        prev_decision = decision
        prev_dF_dTheta = dF_dTheta
        return_list.append(R_t)
    return theta, return_list

def direct_reinforce(price_series, parameters):
    # preprocessing inputs
    input = preprocess(price_series, parameters["preprocess"])
    assert len(price_series) == len(input)

    # divide dataset into training and test set
    input_training = input[:int(len(input) * parameters["training_set"])]
    input_test = input[int(len(input) * parameters["training_set"]):]

    # learning to trade via direct reinforcement
    rl_net_train, returns_train = RL_Trader(input_training, parameters, np.zeros(parameters["time_interval"] + 2))
    rl_net_test, returns_test = RL_Trader(input_test, parameters, rl_net_train)
    return returns_test




def search_opt_aim(parameters):
    PARAMETER["learning_rate"] = parameters["learning_rate"]
    PARAMETER["time_interval"] = parameters["time_interval"] + 5
    PARAMETER["eta"] = parameters["eta"]
    return -1 * sum(direct_reinforce(aim_stock, PARAMETER))



def search_opt(parameters):
    space = {
        "learning_rate":  hp.uniform("learning_rate", 2e-1, 7e-1),
        "time_interval": hp.randint("time_interval", 5),
        "eta": hp.uniform("eta", 2e-1, 7e-1)
        }
    
    parameters["learning_rate"] = space["learning_rate"]
    parameters["time_interval"] = space["time_interval"]
    parameters["eta"] = space["eta"]
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(search_opt_aim , space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None)
    return best


if __name__ == "__main__":
    
    tickers = ['^IXIC', '^GSPC', '^DJI']

    start_date = '2010-06-15'
    end_date = '2019-06-15'
    datas = data.DataReader(tickers, 'yahoo', start_date, end_date)

    close = datas['Close']
    data_all = pd.date_range(start=start_date, end=end_date, freq='B')
    close = close.reindex(data_all)

    close = close.fillna(method='ffill')

    nasdaq = close.loc[:, '^IXIC']
    sp = close.loc[:, '^GSPC']
    dow = close.loc[:, '^DJI']
    
    # choose an aim
    aim_stock = nasdaq


    # try direct reinforcement
    returns_test = direct_reinforce(aim_stock, PARAMETER)
    return_buy_and_hold = np.log(aim_stock[-1]) - np.log(aim_stock[int(len(aim_stock) * PARAMETER["training_set"])])
    print("Abnormal return is: " + str(sum(returns_test) - return_buy_and_hold))
    # optimizing hyper-parameters using hyperopt
    best = search_opt(PARAMETER)
    print(str(best))
