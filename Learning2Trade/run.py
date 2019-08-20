# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:02:11 2019

@author: Luofeng
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


eta = 0.02




def DirectRL(time_interval, prices, dd, rho, mu, commission, w):
    # time_interval: time interval
    # prices: list of the price used
    # dd: Boolean indicating Sharpe or Downside Deviation
    # rho : Optimization step
    # mu : position size
    # commission : comission
    # w first to make a decision
    
    # initialization
    trades = np.array(0)
    ft = 0
    price_array =  np.array(prices[0:time_interval]) 
    returns = price_array[1:time_interval-1] - price_array[:time_interval-2] 
    A_0 = np.mean(returns)
    B_0 = returns.dot(returns) / len(returns)
    DD_0 = np.sqrt(np.where(returns > 0, returns, 0).dot(np.where(returns > 0, returns, 0)) / len(returns))
    AA_0 = np.mean(np.where(returns < A_0, returns, 0))
    AAA_0 = np.mean(np.where(returns < AA_0, returns, 0))
    As = A_0
    Bs = B_0
    DDs = DD_0
    AAs = AA_0
    AAAs = AAA_0
#    returns = price_array[1:time_interval-1] / price_array[:time_interval-2] -1
    if dd == True:
        ddr = ddRatio(returns[-1], As, DDs)
    elif dd == False:
        ddr = Sharpe(returns[-1], As, Bs)
#        ddr = othercriteria(returns, As, Bs, AAs, AAAs)
    # first-order derivative
    x = np.append(returns,ft)
    x = np.append(x,ddr)
    x = np.append(x,1)
    deriv = (1 - np.tanh(w.dot(x))**2)*(x + w)
    derivatives = np.array([deriv,deriv])
    
    returns1 = returns
    commissiones = price_array[-1]*commission
    w1 = w
    
    # return de la inversión
    rend_trader = prices[time_interval]
    capital = np.array(rend_trader)
    price_prev = prices[time_interval]
    for price in prices[time_interval+1:]:
        
        if dd == True:
            ddr = ddRatio(price - price_prev, As, DDs)
        elif dd == False:
            ddr = Sharpe(price - price_prev, As, Bs)
    #        ddr = othercriteria(price - price_prev, As, Bs, AAs, AAAs)
        


        dict_val = Trader(ft,ddr,returns,derivatives,rho,mu,commissiones,w,w1)
        
        ft1 = ft
        ft = dict_val["decision"]
        
        trades = np.append(trades,ft)
        
        returns1 = returns
        price_array = np.delete(price_array,0)
        price_array = np.append(price_array,price)
        returns = price_array[1:time_interval-1] - price_array[:time_interval-2]
#        returns = price_array[1:time_interval-1] / price_array[:time_interval-2] - 1
        
        if dd == True:
            ddr = ddRatio(price - price_prev, As, DDs)
        elif dd == False:
            ddr = Sharpe(price - price_prev, As, Bs)
    #        ddr = othercriteria(price - price_prev, As, Bs, AAs, AAAs)
        
        derivatives = dict_val["derivatives"]
        commissiones = price * commission
        w1 = w
        w = dict_val["weight"]
        
        rend_trader = rend_trader + mu*(ft1*returns1[-1] - commissiones*(ft-ft1))
        capital = np.append(capital,rend_trader)
        
        As = update_para(As, price - price_prev, eta)
        AAs = update_para1(AAs, price - price_prev, As, eta)
        AAAs = update_para1(AAAs,price - price_prev, AAs, eta)
        DDs = DDs + eta * (pow(min(price - price_prev, 0), 2) - DDs)
        Bs = update_para(Bs, pow(price_prev - price, 2), eta)
        price_prev = price

    
    plt.plot(capital, 'b')
    plt.plot(np.array(prices[time_interval:]), 'r')
#    fig.tight_layout()
    plt.show()

    ## compute the return series
    capital_rtn = capital[1:]/capital[:-1] - 1.0
    print('Length of capital_rtn is: %d\n' % len(capital_rtn))
    trading_sr = Sharpe_old(capital_rtn)*np.sqrt(252.0)
    trading_ddratio = ddRatio_old(capital_rtn)
    print("Sharpe ratio of RL trader is: %5.3f, ddRatio is: %5.3f\n" % (trading_sr, trading_ddratio))

    outsample = np.array(prices[time_interval:])
    BH_rtn = outsample[1:]/outsample[:-1] - 1.0
    #print(BH_rtn[:10])
    print('Length of buy_hold_rtn is: %d\n' % len(BH_rtn))
    BH_sr = Sharpe_old(BH_rtn)*np.sqrt(252.0)
    BH_ddratio = ddRatio_old(BH_rtn)
    print("Sharpe ratio of Buy-and-Hold is: %5.3f, ddRatio is: %5.3f\n" % (BH_sr, BH_ddratio))
    plt.plot(capital_rtn, 'g')
    #plt.plot(np.array(prices[time_interval:]), 'r')
#    fig.tight_layout()
    plt.show()
   
    return w, trading_sr, trading_ddratio, BH_sr, BH_ddratio, capital_rtn, BH_rtn
    
    
def Trader(decision,dd,returns,derivatives,rho,mu,commission,w,w1):
    
    x = np.append(returns,decision)
    x = np.append(x,dd)
    x = np.append(x,1)
    ft = round(np.tanh(w.dot(x)))
    
    x1 = x
    
    dF2 = derivatives[-2]
    dF1 = (1 - np.tanh(w.dot(x))**2)*(x1 + w1*dF2)
    dF = (1 - np.tanh(w.dot(x))**2)*(x + w*dF1)
    
    delta = mu*(commission*np.sign(ft-decision)*(dF1 - dF) + returns[-1]*dF1)
    w = w - rho*delta
    
    # Actualizando las derivatives
    derivatives = np.append(derivatives,dF)
    
    dict_val = dict();
    dict_val["decision"] = ft
    dict_val["weight"] = w
    dict_val["derivatives"] = derivatives
    
    return dict_val

def ddRatio(returns, As, DDs):
    if returns > 0:
        return (returns - 0.5 * As) / np.sqrt(DDs)
    return (DDs * (returns - 0.5 * As) - 0.5 * As * pow(returns, 2) ) / np.sqrt(pow(DDs,3))

def Sharpe(returns, As, Bs):
    return (Bs - (returns - As) - 0.5 * As * (pow(returns, 2) - Bs))/ pow(np.sqrt(Bs - pow(As, 2)), 3)

def othercriteria(returns, As, Bs, AAs, AAAs):
    At = update_para(As, returns, eta)
    AAt = update_para1(AAs, returns, At, eta)
    delAAA = AAt - AAAs
    return (Bs * delAAA - 0.5 * AAAs * (pow(returns, 2) - Bs))/ pow(np.sqrt(Bs - pow(As, 2)), 3)

def update_para(paralast, newinfo, eta):
    return paralast + eta * (newinfo - paralast)

def update_para1(paralast, newinfo, benchmark, eta):
    if newinfo < benchmark:
        return paralast + eta * (newinfo - paralast)
    return paralast


def ddRatio_old(returnlist):
    
    means = np.mean(returnlist)
    dstds = np.std(np.where(returnlist < 0, 0, returnlist))
    
    return means/dstds

def Sharpe_old(returnlist):
    means = np.mean(returnlist)
    stds = np.std(np.where(returnlist == 0, 0, returnlist))
    
    return means/stds   


if __name__ == "__main__":
    tickers = ['^IXIC', '^GSPC', '^DJI']
    
    start_date = '2005-06-15'
    end_date = '2019-06-15'
    datas = data.DataReader(tickers, 'yahoo', start_date, end_date)
    
    close = datas['Close']
    data_all = pd.date_range(start=start_date, end=end_date, freq='B')
    close = close.reindex(data_all)
    
    close = close.fillna(method='ffill')
    
    nasdaq = close.loc[:, '^IXIC']
    sp = close.loc[:, '^GSPC']
    dow = close.loc[:, '^DJI']
    
    fig, ax = plt.subplots(3,1,figsize=(20, 10))
    
    ax[0].plot(nasdaq.index, nasdaq)
    ax[0].set_ylabel('NASDAQ 100', fontsize = 25)
    
    ax[1].plot(sp.index, sp)
    ax[1].set_ylabel('S&P 500', fontsize = 25)
    
    ax[2].plot(dow.index, dow)
    ax[2].set_ylabel('DJIA 30', fontsize = 25)
    
    fig.tight_layout()
    plt.show()
    
    time_interval = 20
    dd = False  # Objective: Sharpe ratio
    #dd = True    # Objective: return/semi-deviation
    rho = 0.025
    mu = 1
    commission = 0.005
    sample_days = 1700
    prices_entrenamiento = nasdaq[:sample_days] #nasdaq[:1504]
    prices_prueba = nasdaq[sample_days:]    #nasdaq[1504:]
    w = np.zeros(time_interval+1)
    
    # In-sample performance
    print ("Prueba sin dd Ratio \n")
    ws,trading_sr, trading_ddratio, BH_sr, BH_ddratio, capital_rtn, BH_rtn = DirectRL(time_interval, prices_entrenamiento, dd, rho, mu, commission,w) # Entrenamiento
    
    ## Out-of-sample performance
    DirectRL(time_interval, prices_prueba, dd, rho, mu, commission, ws)     # Prueba
