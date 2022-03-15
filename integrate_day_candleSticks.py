import numpy as np
import pandas as pd
import datetime as dt
import time
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


def integrated_data(time_interval):
    """
    integrates and saves data that create by function "convert2CandleSticks", each csv file contains candlesticks within
    a day with specific time interval
    :param time_interval: time interval of candlesticks
    :param save_path: save path
    """
    # read_path
    read_dir = './clean_data/Tst'
    # time_intervals = 3600
    read_postfix = 'candlesticks' + str(time_interval) + '.csv'
    except_file_data = [
        '2022-01-08',
        '2022-01-09',
        '2022-01-15',
        '2022-01-16',
        '2022-01-22',
        '2022-01-23',
        '2022-01-29',
        '2022-01-30',
        '2022-02-05',
        '2022-02-06',
        '2022-02-12',
        '2022-02-13',
        '2022-02-19',
        '2022-02-20',
        '2022-02-26',
        '2022-02-27',
        '2022-02-28']
    except_file_data = pd.to_datetime(except_file_data).date
    dti = pd.date_range("2022-01-04", periods=60, freq="D")
    dti = pd.to_datetime(dti).date
    dti = np.setdiff1d(dti, except_file_data)  # remove except_file_data

    date_sets = [date for date in dti]
    read_path_sets = [read_dir + str(date) + read_postfix for date in dti]

    # save integrated data
    integrated_data = pd.DataFrame(columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close'])
    for i, file_path in enumerate(read_path_sets):
        df = pd.read_csv(file_path)
        time_off_set = 60 * 60 * 8  # +8:00
        df['Time'] = df['Time'].apply(
            lambda x: time.strftime("%H:%M", time.gmtime(x + time_off_set)))  # convert second to time
        df['Date'] = date_sets[i]
        integrated_data = pd.concat([integrated_data, df[1:]], ignore_index=True)
    # save data
    save_dir = "./clean_data/"
    save_name_1 = "integrated_"
    save_name_2 = "candlesticks" + str(time_interval) + ".csv"
    save_path = save_dir + save_name_1 + save_name_2
    integrated_data.to_csv(save_path, index=False)


integrated_data(900)

#  plot integrated close price
# close_price_series = pd.DataFrame(columns=["Date", "Time", "Close"])
# for i, file_path in enumerate(read_path_sets):
#     df = pd.read_csv(file_path)
#     tmp = df[["Time", "Close"]]
#     tmp["Date"] = date_sets[i]
#     close_price_series = pd.concat([close_price_series, tmp[1:]], ignore_index=True)
#
# price = close_price_series.Close
# data = close_price_series.Date
# x = np.arange(0, len(price) + 1, 5000)
# plt.xticks(x, data[x])
# plt.plot(price)
# plt.show()
