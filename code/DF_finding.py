#coding:utf-8

import pandas as pd
import numpy as np

PCA_count = 15
def ids_prose(ls):
    ls[1] = '-'.join(ls[1].split('.')[:2])
    return ls
def log_reader(log_path):
    train_log = log_path
    train_log = open(train_log)
    train_log.readline()
    train_log = [ids_prose(x.split()) for x in train_log.readlines()]
    return train_log
def log2data(log1, log2, log_set, usr_set):
    data = np.zeros((len(usr_set),len(log_set)))
    for x in log1:
        data[usr_set[x[0]]][log_set[x[1]]]+=1
    for x in log2:
        data[usr_set[x[0]]][log_set[x[1]]]+=1
    return data
train_log = log_reader('../data/train_log.csv')
test_log = log_reader('../data/test_log.csv')
log = []
log.extend(train_log)
log.extend(test_log)
log_set = {}
usr_set = {}
usr_list = []
for x in log:
    if not x[1] in log_set:
        log_set[x[1]]=len(log_set)
    if not x[0] in usr_set:
        usr_list.append(x[0])
        usr_set[x[0]]=len(usr_set)
data = log2data(train_log, test_log, log_set, usr_set)
from sklearn.decomposition import PCA
pca = PCA(n_components=PCA_count,whiten=True)
data_t = pca.fit_transform(data)
t = pd.DataFrame(data_t, index = usr_list, columns =['pca_'+str(x) for x in list(range(1,PCA_count+1))]).reset_index()
log_type = t.rename(columns={'index':'USRID'})

train_agg = pd.read_csv('../data/train_agg.csv',sep='\t',dtype={'USRID':object})
test_agg = pd.read_csv('../data/test_agg.csv',sep='\t',dtype={'USRID':object})
agg = pd.concat([train_agg,test_agg],copy=False)

# 日志信息
train_log = pd.read_csv('../data/train_log.csv',sep='\t',dtype={'USRID':object})
test_log = pd.read_csv('../data/test_log.csv',sep='\t',dtype={'USRID':object})
log = pd.concat([train_log,test_log],copy=False)

log = log.groupby(['USRID']).size().reset_index(name='counts')
log = pd.merge(log,log_type,on=['USRID'],how='left',copy=True)
# 用户唯一标识
train_flg = pd.read_csv('../data/train_flg.csv',sep='\t',dtype={'USRID':object})
test_flg = pd.read_csv('../data/submit_sample.csv',sep='\t',dtype={'USRID':object})
test_flg['FLAG'] = -1
del test_flg['RST']
flg = pd.concat([train_flg,test_flg],copy=True)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=True)
# raise ValueError
import time
# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
# 这个部分可以发掘的特征其实很多很多很多很多
# log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
# log = log.sort_values(['USRID','OCC_TIM'])
# log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

# log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
#     'next_time_mean':np.mean,
#     'next_time_std':np.std,
#     'next_time_min':np.min,
#     'next_time_max':np.max
# })

#data = pd.merge(data,log,on=['USRID'],how='left',copy=False)
#data.where(data.notnull(), 0)
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

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []
#for skf in [skf1,skf2,skf3]:
for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'is_unbalance':'true',
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

