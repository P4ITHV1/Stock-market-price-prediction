from nsepy import get_history
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensforflow.keras.layers import LSTM



inp = input("type code: ")

data = get_history(symbol=inp, start=date(2015,1,1), end=date(2015,1,31))
data.to_csv("data.csv")
df = pd.read_csv("data.csv")

df1 = df.reset_index()['close']

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset,time_stamp=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_stamp-1):
        a = dataset[i:(i+time_stamp),0]
        dataX.append(a)
        dataY.append(dataset[i+time_stamp, 0])
    return np.array(dataX) , np.array(dataY)

time_step = 100
X_train , y_train = create_dataset(train_data,time_step)
X_test , y_test = create_dataset(test_data , time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1], 1)



