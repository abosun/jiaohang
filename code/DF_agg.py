#coding:utf-8

import pandas as pd
import numpy as np

PCA_count = 15


train_agg = pd.read_csv('../data/train_agg.csv',sep='\t',dtype={'USRID':object})
test_agg = pd.read_csv('../data/test_agg.csv',sep='\t',dtype={'USRID':object})
agg = pd.concat([train_agg,test_agg],copy=False)

# 用户唯一标识
train_flg = pd.read_csv('../data/train_flg.csv',sep='\t',dtype={'USRID':object})
test_flg = pd.read_csv('../data/submit_sample.csv',sep='\t',dtype={'USRID':object})
test_flg['FLAG'] = -1
del test_flg['RST']
flg = pd.concat([train_flg,test_flg],copy=True)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=True)
# raise ValueError
# raise ValueError
import time
# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
# 这个部分可以发掘的特征其实很多很多很多很多
data = data.fillna(0)
from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')

y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values
N = 5
skf = StratifiedKFold(n_splits=N,shuffle=True,random_state=40)
#skf1 = StratifiedKFold(n_splits=3,shuffle=True,random_state=40)
#skf2 = StratifiedKFold(n_splits=4,shuffle=True,random_state=40)
#skf3 = StratifiedKFold(n_splits=5,shuffle=True,random_state=40)

#import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []
#for skf in [skf1,skf2,skf3]:
for train_in,test_in in skf.split(X,y):
    import lightgbm as lgb
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'num_threads':16,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0
for i in xx_pre:
    s = s + i

s = s /len(xx_pre)
res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(s)

print('xx_cv',np.mean(xx_cv))

import time
time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('../submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')

