import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime
from pmdarima.arima import auto_arima
from pmdarima.arima import ARIMA

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# import data
# import "USDJPY"
# data_dir = "./data"
# data_name = "USDJPY30.csv"
# data_path = data_dir + "/" + data_name
# df = pd.read_csv(data_path, header=None, parse_dates=True)
# df = df.drop(labels=range(3000, df.shape[0]), axis=0)
# df[0] = df[0].str.replace('.', '-')
# df[1] = df[1].astype(str) + ':00'
# df[0] = df[0] + ' ' + df[1]  # combine two cols
# df[0] = pd.to_datetime(df[0])
# df = df.drop(columns=[1])  # remove column 1
# df = df.set_index([0])
# df = df.rename(columns={0: 'Time', 2: 'Open', 3: 'High', 4: 'Low', 5: 'Close', 6: 'Volume'})
# # data = smooth(df.Close, 1)
# # data = data[:-1]
# data = df.Close


# import data
# import HSBC data
data = pd.read_csv("./clean_data/integrated_candlesticks3600.csv")
data = data['Close']
# data = data['Close'].rolling(window=60).mean()
# data = data[60:]

# train test split
n = int(len(data) * 0.8)
train = np.array(data[:n])
test = np.array(data[n:])

# auto arima
model = auto_arima(train, trace=True,
                   start_p=0, d=None, start_q=0,
                   max_p=20, max_q=20, max_order=None,
                   information_criterion="aic", stepwise=True)
print(model.summary())


# simple arima
# model = ARIMA(order=(1, 1, 1)).fit(train)
# print(model.summary())
# model.fit(train)
# print(model.summary())


# predict
# pred, conf_in = model.predict_in_sample(return_conf_int=True)
#
# # plot in sample predict
# fig, ax = plt.subplots()
# ax.set(title='Price', xlabel='Date', ylabel='Price')
# x = range(1,len(train))
# plt.plot(train[1:])
# plt.plot(pred[1:])
# ax.fill_between(x, conf_in[1:, 0], conf_in[1:, 1], color='b', alpha=.1)
# legend = ax.legend(loc='lower right')
# plt.show()

pred, conf_in = model.predict(n_periods=20, return_conf_int=True)
fig, ax = plt.subplots()
ax.set(title='Price', xlabel='Date', ylabel='Price')
x = range(len(pred))
plt.plot(x, test[:len(pred)])
plt.plot(x, pred)
ax.fill_between(x, conf_in[:, 0], conf_in[:, 1], color='b', alpha=.1)
plt.show()

# n_step = 10
# pred_series = np.zeros((len(test), n_step))
# conf_in_series = np.zeros((len(test), n_step, 2))
# # predict future
# for i in range(len(test)):
#     tmp = test[:i]
#     observed = np.append(train, test[:i])
#     pred, conf_in = model.fit_predict(observed, n_periods=n_step, return_conf_int=True)
#     pred_series[i] = pred
#     # conf_in_series[i,:,:] = conf_in.flatten()
# # plot prediction
# fig, ax = plt.subplots()
# ax.set(title='Price', xlabel='Date', ylabel='Price')
# x = range(len(test))
# plt.plot(x, test)
# for i in range(len(test)):
#     plt.plot(range(i, i + n_step), pred_series[i])
# # ax.fill_between(x, conf_in_series[:, 0], conf_in_series[:, 1], color='b', alpha=.1)
# legend = ax.legend(loc='lower right')
# plt.show()
