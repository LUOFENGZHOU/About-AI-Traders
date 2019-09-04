# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:29:05 2019

@author: Luofeng
"""

import numpy as np
import math


def preprocess(series, method):
    series = np.array(series)
    if method == "log_dif":
#        print("Preprocessing price to log-price change...")
        return np.append([0], np.log(series)[1:] - np.log(series)[:-1])
    elif method == "dif":
#        print("Preprocessing price to price change...")
        return np.append([0], series[1:] - series[:-1])
    elif method == "n":
        return series
    else:
#        print("Method not understood!")
        return series
    
def update_para(paralast, newinfo, eta):
    return paralast + eta * (newinfo - paralast)


def update_para1(paralast, newinfo, benchmark, eta):
    if newinfo < benchmark:
        return paralast + eta * (newinfo - paralast)
    return paralast

def update_theta(theta, data_in, rt, parameters, hat_list, prev_dF_dTheta=False, prev_decision=1):
    # get the present decision first
    decision = round(np.tanh(theta.dot(data_in)))

    # update theta through chain rule
    dF_dTheta = (1 - pow(np.tanh(theta.dot(data_in)), 2)) * data_in
    if decision < prev_decision:
        dR_dFt = parameters["position_size"] * parameters["commission"]
        dR_dFt_1 = parameters["position_size"] * (rt - parameters["commission"])
    elif decision > prev_decision:
        dR_dFt = -1 * parameters["position_size"] * parameters["commission"]
        dR_dFt_1 = parameters["position_size"] * (rt + parameters["commission"])
    else:  # when Ft = Ft_1, the diff of abs() is not defined, just treat it as zero
        dR_dFt = 0
        dR_dFt_1 = parameters["position_size"] * rt

    R_t = parameters["position_size"] * (
            prev_decision * rt - parameters["commission"] * abs(prev_decision - decision))

    dD_dR = get_diff_criteria(R_t, hat_list, parameters)

    theta_new = theta + parameters["learning_rate"] * dD_dR * (dR_dFt * dF_dTheta + dR_dFt_1 * prev_dF_dTheta)
    return theta_new, decision, dF_dTheta, R_t


def get_hat_list(input, parameters, inputlist=False):
    if inputlist == False:  # initialization
        if parameters["criteria"] == "sharpe":
            return [np.mean(input), np.mean(input.dot(input))]
        elif parameters["criteria"] == "downside_ratio":
            return [np.mean(input), input.dot(input) / len(input)]
        elif parameters["criteria"] == "downside_risk":
            first_order = np.mean(input)
            first_order2 = np.sum(np.where(input > first_order, 0, input)) / len(input) * 2
            first_order3 = np.sum(np.where(input > first_order2, 0, input)) / len(input) * 4
            return [first_order, first_order2, first_order3, input.dot(input) / len(input)]
        else:
            print("Criteria not understood!")
            return None
    else:  # updating
        if parameters["criteria"] == "sharpe":
            return [update_para(inputlist[0], input, parameters["eta"]),
                    update_para(inputlist[1], pow(input,2), parameters["eta"])]
        elif parameters["criteria"] == "downside_ratio":
            return [update_para(inputlist[0], input, parameters["eta"]),
                    inputlist[1] + parameters["eta"] * (pow(min(input, 0), 2) - inputlist[1])]
        elif parameters["criteria"] == "downside_risk":
            first_order = update_para(inputlist[0], input, parameters["eta"])
            first_order2 = update_para1(inputlist[1], input, first_order, parameters["eta"])
            first_order3 = update_para1(inputlist[2], input, first_order2, parameters["eta"])
            return [first_order, first_order2, first_order3,
                    inputlist[3] + parameters["eta"] * (pow(min(input, 0), 2) - inputlist[3])]

def get_diff_criteria(Rt, hat_list, parameters):
    if parameters["criteria"] == "sharpe":
        return  (hat_list[1] - hat_list[0] * Rt) / math.sqrt((hat_list[1] - hat_list[0]**2)** 3)
    elif parameters["criteria"] == "downside_ratio":
        if Rt > 0:
            return 1 / np.sqrt(hat_list[1])
        else:
            return 1 / hat_list[1] - Rt * hat_list[0] / math.sqrt(hat_list[1] ** 3)
    elif parameters["criteria"] == "downside_risk":
        if Rt > hat_list[0] and Rt > 0:
            return 0
        elif Rt > hat_list[0] and Rt <= 0:
            return Rt * hat_list[1] / math.sqrt(hat_list[3]**3)
        elif Rt <= hat_list[0] and Rt > 0:
            return 1 / np.sqrt(hat_list[3])
        else:
            return 1 / hat_list[3] - Rt * hat_list[1] / math.sqrt(hat_list[3]** 3)


def sharpe(returnlist):
    # calculate sharpe ratio
    means = np.mean(returnlist)
    stds = np.std(returnlist)
    return means / stds


def downside_ratio(returnlist):
    means = np.mean(returnlist)
    downside_std = np.std(np.where(returnlist < 0, 0, returnlist))
    return means / downside_std

