#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:40:06 2020

@author: Manting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

def fillup(df):
    #print(df)
    check = ['#','*','x','A']
    change = []
    for i in range(3,len(df)):
        right = i + 1
        if any(x in df[i] for x in check):
            change.append(int(i))
            #print(change)
            for k in range(i + 1, len(df)):
                #print(k)
                if i == 3:
                    df[i] = df[k]
                    right = k
                    break
                if any(x in df[k] for x in check):
                    right = k + 1
                else:
                    break
            
            for j in range(i - 1, 2, -1):
                #print(right)
                if right >= len(df):
                    df[i]  = df[j]
                    break
                if j not in change:
                    #print(df[j],df[right])
                    df[i] = str((float(df[j])+ float(df[right])) / 2)
                    break                   
    #print(df)

def transdata(X):
    train = []
    listX = [X[x : x + 18] for x in range(0, len(X), 18)]
    for i in range(len(listX)):
        numpy_array = np.array(listX[i])ee 
        transpose = numpy_array.T
        transpose_list = transpose.tolist()
        train += transpose_list[3:]
    numpy_array = np.array(train)
    transpose = numpy_array.T
    return transpose.tolist()
    #print(len(train))
    
def pm(train, test, size, index):
    X_train = []
    y_train = []
    for i in range(len(train[9]) - index):
        X_train.append(train[9][i : i + size])
        y_train.append(train[9][i + index])
    X_test = []
    y_test = []
    for i in range(len(test[9]) - index):
        X_test.append(test[9][i : i + size])
        y_test.append(test[9][i + index])
    
    X_train = np.array(X_train,dtype = np.float32)
    y_train = np.array(y_train,dtype = np.float32)
    X_test = np.array(X_test,dtype = np.float32)
    y_test = np.array(y_test,dtype = np.float32)
    
    # Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    y1_test_pred = reg.predict(X_test)
    
    # Random Forest Regression
    reg1 = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(X_train, y_train)
    y2_test_pred = reg1.predict(X_test)
    
    # 計算MAE
    mae = mean_absolute_error(y_test, y1_test_pred)
    mae1 = mean_absolute_error(y_test, y2_test_pred)
    print(' Linear Regression= %s \n Random Forest Regression = %s' % (mae, mae1))
    
def label(train, test, size, index):
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()
    for j in range(len(train)):
        a = []
        ddf = pd.DataFrame()
        b = []
        ddf1 = pd.DataFrame()
        for i in range(len(train[j]) - index):
            a.append(train[j][i : i + size])
            ddf = pd.DataFrame(a) 
            b.append(train[j][i + index])
            ddf1 = pd.DataFrame(b)
        X_train = pd.concat([X_train, ddf], axis=1)
        y_train = pd.concat([y_train, ddf1], axis=1)
        
        c = []
        ddf2 = pd.DataFrame()
        d = []
        ddf3 = pd.DataFrame()
        for i in range(len(test[j]) - index):
            c.append(test[j][i : i + size])
            ddf2 = pd.DataFrame(c) 
            d.append(test[j][i + index])
            ddf3 = pd.DataFrame(d)
        X_test = pd.concat([X_test, ddf2], axis=1)
        y_test = pd.concat([y_test, ddf3], axis=1)
    
    # Linear Regression
    reg = LinearRegression().fit(X_train, y_train)
    y1_test_pred = reg.predict(X_test)
    
    # Random Forest Regression
    reg1 = RandomForestRegressor(n_estimators = 100, random_state = 0).fit(X_train, y_train)
    y2_test_pred = reg1.predict(X_test)
    
    # 計算MAE
    mae = mean_absolute_error(y_test, y1_test_pred)
    mae1 = mean_absolute_error(y_test, y2_test_pred)
    print(' Linear Regression= %s \n Random Forest Regression = %s' % (mae, mae1))


# 讀資料
df = pd.read_csv('新竹_2019.csv', encoding ='big5').replace("NA","0")
feature_name = df['測項                  '].iloc[1:][0:18].values.tolist()
#print(feature_name[9])
df.fillna('#', inplace = True) # 空直補0
total = df.iloc[4825 : , : ].values.tolist()
        
# 缺失值以及無效值以前後一小時平均值取代
for i in range(len(total)):
    fillup(total[i])
    #print(total[i])
    
# 10和11月資料當作訓練集X，12月之資料當作測試集y，轉成維度為(18,1464)
train = transdata(total[ : 1098][:])
test = transdata(total[ 1098:][:])

# pm2.5
print('pm2.5 將未來第一個小時當預測目標')
pm(train, test, 6, 6)
print('pm2.5 將未來第六個小時當預測目標')
pm(train, test, 6, 11)

# 所有18種屬性
print('所有屬性將未來第一個小時當預測目標')
label(train, test, 6, 6)
print('所有屬性將未來第六個小時當預測目標')
label(train, test, 6, 11)




