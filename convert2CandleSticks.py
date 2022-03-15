import numpy as np
import pandas as pd


def conv2CandleStcks(read_path, save_path, time_intervals):
    # import data
    # data_dir = "./"
    # data_name_1 = "Tst2022-01-04"
    # data_name_2 = "tapes.csv"
    # data_path = data_dir + "/" + data_name_1 + data_name_2
    data_path = read_path
    df = pd.read_csv(data_path, header=None, parse_dates=True)
    time = df[2].values
    # int_time = time.astype(int)
    piece = df[3].values

    total_points = len(df[2])
    # time_intervals = 60*15  # second
    int_time = (time / time_intervals).astype(int)
    open_time = 0  # 8am – 4.30pm
    close_time = 30600

    # define cs df
    cs_df = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close'])
    cs_df = cs_df.append({'Time': 0, 'Open': 0, 'High': 0, 'Low': 0, 'Close': 0}, ignore_index=True)

    # convert to candlesticks
    last_id = 0
    for t in range(open_time, close_time, time_intervals):
        piece_in_intervals = []

        # if exist records
        if int_time[last_id] == int(t / time_intervals):
            for i in range(last_id, total_points):
                if int_time[last_id] == int(t / time_intervals):
                    piece_in_intervals.append(piece[last_id])
                    # last sample
                    if last_id == total_points - 1:
                        cs_df = cs_df.append({'Time': t+time_intervals, 'Open': piece_in_intervals[0], 'High': max(piece_in_intervals),
                                              'Low': min(piece_in_intervals), 'Close': piece_in_intervals[-1]},
                                             ignore_index=True)
                        break
                    else:
                        last_id = last_id + 1
                else:
                    cs_df = cs_df.append({'Time': t+time_intervals, 'Open': piece_in_intervals[0], 'High': max(piece_in_intervals),
                                          'Low': min(piece_in_intervals), 'Close': piece_in_intervals[-1]},
                                         ignore_index=True)
                    break
        # if not exist records
        else:
            last_row = cs_df.tail(1)
            last_row_value = last_row.values
            cs_df = cs_df.append({'Time': t, 'Open': last_row_value[0, 1], 'High': last_row_value[0, 2],
                                  'Low': last_row_value[0, 3], 'Close': last_row_value[0, 4]}, ignore_index=True)
            continue

    # save data
    # save_dir = "./"
    # save_name_1 = data_name_1
    # save_name_2 = "candlesticks.csv"
    # save_path = save_dir + save_name_1 + save_name_2
    save_path = save_path
    cs_df.to_csv(save_path, index=False)


# read_path
read_dir = "./data/Tst"
save_dir = './clean_data/Tst'
read_postfix = 'tapes.csv'
time_intervals = 15*60
save_postfix = 'candlesticks'+str(time_intervals)+'.csv'
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

read_path_sets = [read_dir + str(date) + read_postfix for date in dti]
save_path_sets = [save_dir + str(date) + save_postfix for date in dti]

for index in range(len(read_path_sets)):
    conv2CandleStcks(read_path_sets[index], save_path_sets[index], time_intervals)
