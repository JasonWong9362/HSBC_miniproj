import pandas as pd
import numpy as np
import pandas as pd
import math
import csv
import itertools
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
from tqdm import tqdm

# import data
data_dir = "./data"
data_name = "USDJPY30.csv"
data_path = data_dir + "/" + data_name
df = pd.read_csv(data_path, header=None, parse_dates=True)
df = df.drop(labels=range(6000, df.shape[0]), axis=0)
df[0] = df[0].str.replace('.', '-')
df[1] = df[1].astype(str) + ':00'
df[0] = df[0] + ' ' + df[1]  # combine two cols
df[0] = pd.to_datetime(df[0])
df = df.drop(columns=[1])  # remove column 1
df = df.set_index([0])
df = df.rename(columns={0: 'Time', 2: 'Open', 3: 'High', 4: 'Low', 5: 'Close', 6: 'Volume'})


# print(df)

def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1 / n, adjust=False).mean()


def atr_(df, period):
    data = df.df.copy()
    data_len = data.shape[0]
    high = data['High']
    low = data['Low']
    close = data['Close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)

    atr = np.zeros(data_len)
    atr[0:period] = 0
    atr[period] = sum(tr[1:period + 1]) / period
    for i in range(period + 1, data_len):
        atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period
    # ATRBuffer_rma[i] = (ATRBuffer[i - 1] * (InpAtrPeriod - 1) + TRBuffer[i]) / InpAtrPeriod;
    # atr = wwma(tr, n)
    return atr


def choppiness_index(df, period):
    data_len = df.df.shape[0]
    chop = np.zeros(data_len)
    chop[0:period - 1] = 0
    atr = atr_(df, 1)
    for i in range(period, data_len):
        SUM_ATR = np.sum(atr[i - period + 1:i])
        MaxHi = np.max(df['High'][i - period + 1:i])
        MinLo = np.min(df['Low'][i - period + 1:i])
        if MaxHi == MinLo:
            chop[i] = None
        else:
            chop[i] = 100 * math.log(SUM_ATR / (MaxHi - MinLo), 10) / math.log(period, 10)
    return chop


def atr_var(df, period, keyvalue):
    data_len = df.df.shape[0]
    atr_var = np.zeros(data_len)
    atr_var[0:period] = 0
    atr = atr_(df, period)

    for i in range(period, data_len):
        close = df['Close'][i]
        closeprevious = df['Close'][i - 1]
        nloss = keyvalue * atr[i]

        if close > atr_var[i - 1] and closeprevious > atr_var[i - 1]:
            atr_var[i] = max(atr_var[i - 1], (close - nloss))

        elif close < atr_var[i - 1] and closeprevious < atr_var[i - 1]:
            atr_var[i] = min(atr_var[i - 1], (close + nloss))

        elif close > atr_var[i - 1]:
            atr_var[i] = close - nloss

        else:
            atr_var[i] = close + nloss
    return atr_var


# AtrVarCross(period=41,keyvalue=6.6,chop_period=2,chop_value=5)
class AtrVarCross(Strategy):
    # Define the two MA lags as *class variables*
    # for later optimization
    chop_period = 2
    chop_value = 5
    atr_period = 41
    atr_keyvalue = 6.6
    size_ = 0.5

    # def __init__(self, chop_period, chop_value, atr_period, atr_keyvalue, size_):
    #     self.chop_period = chop_period
    #     self.chop_value = chop_value
    #     self.atr_period = atr_period
    #     self.atr_keyvalue = atr_keyvalue
    #     self.size_ = size_

    def init(self):
        # Precompute the two moving averages
        self.atr = self.I(atr_, self.data, self.atr_period)
        self.atr_var = self.I(atr_var, self.data, self.atr_period, self.atr_keyvalue)
        self.chop = self.I(choppiness_index, self.data, self.chop_period)

    def next(self):
        if self.chop[-1] < self.chop_value:
            if self.atr_var[-1] < self.data.Close[-1]:
                if self.position.is_long == 0:
                    if self.position.is_short == 1:
                        self.position.close()
                        self.buy(size=self.size_)
                    else:
                        self.buy(size=self.size_)

            else:
                if self.position.is_short == 0:
                    if self.position.is_long == 1:
                        self.position.close()
                        self.sell(size=self.size_)
                    else:
                        self.sell(size=self.size_)

        else:
            if self.position.is_short == 1 or self.position.is_long == 1:
                self.position.close()


# cartesian product of params

atr_period = np.array(range(1, 100, 10))
atr_keyvalue = np.array([x / 10.0 for x in range(1, 100, 10)])
chop_period = np.array(range(1, 100, 10))
chop_value = np.array(range(1, 100, 10))
total_iter = len(atr_period) * len(atr_keyvalue) * len(chop_period) * len(chop_value)
print("tuning paras combo:  " + str(total_iter))

bt = Backtest(df, AtrVarCross, cash=10_000, commission=0)

# create csv file
csv_dir = './result'
csv_name = data_name
csv_path = csv_dir + "/" + csv_name

# result_name = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
#        'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]',
#        'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio',
#        'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]',
#        'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
#        '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
#        'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
#        'Profit Factor', 'Expectancy [%]', 'SQN', '_strategy', '_equity_curve',
#        '_trades']
result_name = ['Exposure Time [%]', 'Equity Final [$]',
               'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]',
               'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'Sharpe Ratio',
               'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]',
               'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
               '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
               'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
               'Profit Factor', 'Expectancy [%]', 'SQN']
param_name = ['atr_period', 'atr_keyvalue', 'chop_period', 'chop_value']
header = result_name + param_name

# with open(csv_path, 'w') as f:
#     writer = csv.writer(f)
#     # write the header
#     writer.writerow(header)
#     with tqdm(total=total_iter, position=0, leave=True) as pbar:
#         for i in tqdm(itertools.product(atr_period, atr_keyvalue, chop_period, chop_value), position=0, leave=True):
#             pbar.update()
#             try:
#                 stats = bt.run(atr_period=i[0], atr_keyvalue=i[1], chop_period=i[2], chop_value=i[3], size_=0.5)
#                 # write the data
#                 stats_data = stats.values[3:27]
#                 param_data = np.array(i)
#                 data = np.concatenate((stats_data, param_data), axis=None)
#                 writer.writerow(data)
#                 # print("loaded combo:" + str(i))
#             except:
#                 continue

stats = bt.run(atr_period=21, atr_keyvalue=8.1, chop_period=31, chop_value=51, size_=0.5)
print(stats)
bt.plot()

# stats = bt.optimize(atr_period=range(3, 100, 2),
#                     atr_keyvalue=[x / 10.0 for x in range(1, 100, 5)],
#                     chop_period=(3, 100, 2),
#                     chop_value=(1, 100, 5),
#                     maximize='Equity Final [$]')
# print(stats._strategy)
