import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# create samples of LOBs
# data = []
# f = open('Tst2022-01-04LOBs.txt', "r")
# lines = f.readlines()
# data = lines[30000:600000]
# del lines
# with open('LOB_exmaple.txt', 'w') as f:
#     for line in data:
#         f.write(line)

# visualize class distribution
# future_trend = LOBs_array[:-(ks[-1]), -len(ks):].astype(int)
# fig = plt.figure()
# # fig.subplots_adjust(hspace=0.4, wspace=0.4)
# for i in range(len(ks)):
#     ax = fig.add_subplot(2, 3, i + 1)
#     ax.hist(future_trend[:, i])
#     ax.set_title('k =' + str(ks[i]))
# plt.show()


def read_data(data_path, count, ks, up_threshold, down_threshold):
    # count: bid, ask order count
    # ks: predict kth ahead ticks

    # read LOB txt and convert to string in json format
    f = open(data_path, "r")
    lines = f.readlines()
    raw_data = []
    for line_id in tqdm(range(len(lines))):
        tmp = lines[line_id].replace(' [', '').replace(' ]', '')
        tmp = tmp.replace(' ', '').replace('\n', '')
        tmp = tmp.replace('[', '{').replace(']', '}-')
        tmp = tmp.replace('"time",', '"time":').replace('"bid",', '"bid":[').replace('"ask",', '"ask":[')
        if tmp != '':
            raw_data.append(tmp)
    print("preprocess 1 finished.")

    for line_id in tqdm(range(len(raw_data))):
        if raw_data[line_id] == '"ask":[':
            raw_data[line_id - 2] = raw_data[line_id - 2] + ']'
        if raw_data[line_id] == '}-':
            raw_data[line_id - 1] = raw_data[line_id - 1] + ']'
    print("preprocess 2 finished.")

    # list to string
    converted_str = ''.join(str(i) for i in raw_data)
    del raw_data
    split_str = converted_str.split('-')
    print("List2String finished.")

    # string to json
    LOBs_json = []
    for i in tqdm(range(len(split_str))):
        if split_str[i] != '':
            try:
                LOBs_json.append(json.loads(split_str[i]))
            except:
                print("string to json error index:" + str(i))
    del split_str
    print("String2Json finished")

    # json to array
    LOBs_array = []
    # count = 5
    for t in tqdm(LOBs_json):
        bid_price = t["bid"][0:count * 2:2]
        bid_size = t["bid"][1:count * 2:2]
        ask_price = t["ask"][0:count * 2:2]
        ask_size = t["ask"][1:count * 2:2]
        per_simple = ask_size[::-1] + bid_size + ask_price[::-1] + bid_price

        if len(per_simple) != int(count*4):
            tmps = [bid_price, bid_size, ask_price, ask_size]
            for i in range(len(tmps)):
                for j in range(count):
                    if len(tmps[i]) != count:
                        tmps[i].append(0)
                    else:
                        break
            per_simple = tmps[3][::-1] + tmps[1] + tmps[2][::-1] + tmps[0]
        LOBs_array.append(per_simple)  # [::-1] reverse
    del LOBs_json
    print("Json2Array finished.")

    # count len per sample
    LOBs_len = []
    for i in range(len(LOBs_array)):
        LOBs_len.append(len(LOBs_array[i]))
    unique, counts = np.unique(LOBs_len, return_counts=True)
    print(dict(zip(unique, counts)))

    LOBs_array = np.array(LOBs_array)

    # calculate mid price
    LOBs_midprice = [np.mean(t[14:16]) for t in LOBs_array]
    LOBs_array = np.c_[LOBs_array, np.array(LOBs_midprice)]

    # count up, down or stationary
    # ks = [1, 5, 10, 50, 100, 1000]
    # up_threshold = 0.002
    # down_threshold = - 0.002

    LOBs_array = np.c_[LOBs_array, np.zeros((LOBs_array.shape[0], len(ks) * 2))]
    for i, k in enumerate(ks):
        for t_id in tqdm(range(LOBs_array.shape[0] - ks[-1])):
            l = ((1 / k) * (np.sum(LOBs_midprice[t_id + 1:t_id + k + 1]) - k * LOBs_midprice[t_id])) / LOBs_midprice[
                t_id]

            # label
            if l > up_threshold:
                label = 0
            elif l < down_threshold:
                label = 2
            else:
                label = 1

            LOBs_array[t_id, count * 4 + 1 + i] = l
            LOBs_array[t_id, count * 4 + 1 + i + len(ks)] = label

    # normalization
    #  LOBs_array[:, :21] = stats.zscore(LOBs_array[:, :21], axis=1)
    LOBs_array = LOBs_array[:-(ks[-1]), :]

    # LOB
    train_test_split_id = int(LOBs_array.shape[0] * 0.7)
    train_lob = LOBs_array[:train_test_split_id, : count*4]
    test_lob = LOBs_array[train_test_split_id + 1:, : count*4]

    # label
    label = LOBs_array[:, -len(ks):]
    # one hot encode
    onehot_label = np.zeros((label.shape[0], label.shape[1], 3))
    for t in range(label.shape[0]):
        try:
            a = np.array(label[t, :]).astype(int)
            b = np.zeros((a.size, 3))
            b[np.arange(a.size), a] = 1  # np.arrange return seq
            onehot_label[t, :, :] = b
        except:
            print(str(b))

    train_label = onehot_label[:train_test_split_id, :, :]
    test_label = onehot_label[train_test_split_id + 1:, :, :]

    # visualize class distribution
    # future_trend = LOBs_array[:-(ks[-1]), -len(ks):].astype(int)
    # fig = plt.figure()
    # # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # for i in range(len(ks)):
    #     ax = fig.add_subplot(2, 3, i + 1)
    #     ax.hist(future_trend[:, i])
    #     ax.set_title('k =' + str(ks[i]))
    # plt.show()

    return train_lob, test_lob, train_label, test_label

# data_path = "LOB_exmaple.txt"
# count = 5
# ks = [1, 100, 200, 300, 500]
# up_threshold = 0.002
# down_threshold = - 0.002
# train_lob1, test_lob1, train_label1, test_label1 = read_data(data_path, count, ks, up_threshold, down_threshold)
