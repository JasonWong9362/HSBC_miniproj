import tensorflow as tf
import os
import logging
import glob
import argparse
import sys
import time
import pandas as pd
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter
from preprocess import *
from model_gpu import get_model_seq
from read_LOB import read_data

# check gpu
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#     # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#         print(e)

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

# please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'
T = 50
epochs = 50
batch_size = 32
n_hidden = 64
checkpoint_filepath = './model_deeplob_seq/weights'

# dec_train = np.loadtxt('Train_Dst_NoAuction_DecPre_CF_7.txt')
# dec_test1 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
# dec_test2 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_8.txt')
# dec_test3 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_9.txt')
# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
#
# # extract limit order book data from the FI-2010 dataset
# train_lob = prepare_x(dec_train)  # time*40(top10 size price ask bid)
# test_lob = prepare_x(dec_test)
#
# # extract label from the FI-2010 dataset
# train_label = get_label(dec_train)
# test_label = get_label(dec_test)

data_path = "LOB_exmaple.txt"
count = 10
ks = [1, 100, 200, 300, 500]
up_threshold = 0.002
down_threshold = - 0.002
train_lob, test_lob, train_label, test_label = read_data(data_path, count, ks, up_threshold, down_threshold)

# prepare training data. We feed past 100 observations into our algorithms.
train_encoder_input, train_decoder_target = data_classification(train_lob, train_label, T)
train_decoder_input = prepare_decoder_input(train_encoder_input, teacher_forcing=False)

test_encoder_input, test_decoder_target = data_classification(test_lob, test_label, T)
test_decoder_input = prepare_decoder_input(test_encoder_input, teacher_forcing=False)

print(f'train_encoder_input.shape = {train_encoder_input.shape},'
      f'train_decoder_target.shape = {train_decoder_target.shape}')
print(f'test_encoder_input.shape = {test_encoder_input.shape},'
      f'test_decoder_target.shape = {test_decoder_target.shape}')

model = get_model_seq()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

split_train_val = int(np.floor(len(train_encoder_input) * 0.8))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

model.fit([train_encoder_input[:split_train_val], train_decoder_input[:split_train_val]],
          train_decoder_target[:split_train_val],
          validation_data=([train_encoder_input[split_train_val:], train_decoder_input[split_train_val:]],
          train_decoder_target[split_train_val:]),
          epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)
pred = model.predict([test_encoder_input, test_decoder_input])

evaluation_metrics(test_decoder_target, pred)






