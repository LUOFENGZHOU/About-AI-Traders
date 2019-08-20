import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
from scipy.stats import pearsonr
import csv
from hyperopt import fmin, tpe, hp, partial


def find_time_ind(df):
    day_ind = [0]
    day_temp = df["TradingDay"][0]
    for i in range(1, len(df)):
        if df["TradingDay"][i] != day_temp:
            day_ind.append(i)
            day_temp = df["TradingDay"][i]
    return day_ind

# input cell
train = pd.read_csv('0805rolling5.csv')
# train = pd.read_csv('all_data.csv')
# train = pd.read_csv('acou_and_factors.csv')
x_column = ["factor"+str(i)+'_'+str(j) for i in range(0,17) for j in range(5)]
# x_column = ["factor"+str(i) for i in range(0,17)]
for i in range(12):
    for j in range(5):
        x_column.append("10M" + str(i) +'_'+str(j))
# for i in range(12):
#     x_column.append("10M" + str(i))

train = train.sort_values(by = ["TradingDay", "SecuCode"], ascending = [True, True])
train.reset_index(drop=True, inplace=True)
X = train[x_column]
target = ['NextReturnCate']
y = train[target]
y += 1
yp = train["NextReturn"]
time_ind = find_time_ind(train)


# initializing
train_days = 775
delta_days = -776
X_train = X[time_ind[0]:time_ind[train_days]]
X_test = X[time_ind[train_days]:time_ind[train_days+delta_days]]
y_train = y[time_ind[0]:time_ind[train_days]]
y_test = y[time_ind[train_days]:time_ind[train_days+delta_days]]
yp_train = yp[time_ind[0]:time_ind[train_days]]
yp_test = yp[time_ind[train_days]:time_ind[train_days+delta_days]]
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)
ddev = xgb.DMatrix(data=X_test, label=y_test)

evallist = [(ddev, 'eval'), (dtrain, 'train')]



# 自定义hyperopt的参数空间
space = {"max_depth": hp.randint("max_depth", 25),
         "n_estimators": hp.randint("n_estimators", 30),
         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),
         "subsample": hp.randint("subsample", 5),
         "min_child_weight": hp.randint("min_child_weight", 6),
         "alpha": hp.uniform("alpha", 0, 1),
         "lambda": hp.uniform("lambda", 0, 1)
         }

def argsDict_tranform(argsDict, isPrint=False):
    argsDict["max_depth"] = argsDict["max_depth"] + 5    # from 5 to 30
    argsDict['n_estimators'] = argsDict['n_estimators'] + 10  # from 10 to 40
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.001   # from 0.001 to 0.01
    argsDict["subsample"] = argsDict["subsample"] * 0.1 + 0.5  # from 0.5 to 1
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 1 # from 1 to 6
    argsDict["alpha"] = argsDict["alpha"] * 20
    argsDict["lambda"] = argsDict["lambda"] * 50
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict
from sklearn.metrics import mean_squared_error, zero_one_loss

def xgboost_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)
    
    params = {'nthread': -1,  # 进程数
              'max_depth': argsDict['max_depth'],  # 最大深度
              'n_estimators': argsDict['n_estimators'],  # 树的数量
              'eta': argsDict['learning_rate'],  # 学习率
              'subsample': argsDict['subsample'],  # 采样数
              'min_child_weight': argsDict['min_child_weight'],  # 终点节点最小样本占比的和
              'objective': 'reg:squarederror',
              'silent': 0,  # 是否显示
              'gamma': 0,  # 是否后剪枝
              'colsample_bytree': 0.7,  # 样本列采样
              'alpha': argsDict['alpha'],  # L1 正则化
              'lambda': argsDict['lambda'],  # L2 正则化
              'scale_pos_weight': 0,  # 取值>0时,在数据不平衡时有助于收敛
              'seed': 100,  # 随机种子
              'tree_method':'exact'
              }

    xrf = xgb.train(params, dtrain, params['n_estimators'], evallist,early_stopping_rounds=100)

    return get_tranformer_score(xrf)


def get_tranformer_score(tranformer):
    
    xrf = tranformer
    pred = xrf.predict(ddev, ntree_limit=xrf.best_ntree_limit)
  
    return -pearsonr(np.array(yp_test), pred)[0]
# 开始使用hyperopt进行自动调参
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(xgboost_factory, space, algo=algo, max_evals=100, pass_expr_memo_ctrl=None)
dict = best
w = csv.writer(open("0819output.csv", "w"))
for key, val in dict.items():
    w.writerow([key, val])