import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.utils import load_data
import warnings

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', size=None, features='S',
                 target='OT', timeenc=0, freq='h', scale=True, seasonal_patterns=None):
        # size: [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'vali']
        assert features in ['MS', 'S', 'M']
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target  # prediciton target when features is MS or S
        self.timeenc = timeenc
        self.freq = freq
        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border_heads = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border_tails = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border_head = border_heads[self.set_type]
        border_tail = border_tails[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            # 第0列是datetime
            df_data = df_raw[df_raw.columns[1:]]
        else:  # self.features == 'S'
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border_heads[0], border_tail[0]]
            self.scaler.fit(train_data.values)  # norm
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border_head: border_tail]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_x = data[border_head, border_tail]
        self.data_y = data[border_head, border_tail]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.pred_len + self.label_len

        seq_x = self.data_x[s_begin, s_end]
        seq_y = self.data_y[r_begin, r_end]  # y的长度为pred_len + label_len，作用见exp_long_term_forecasting.py line137
        seq_x_mark = self.data_stamp[s_begin, s_end]
        seq_y_mark = self.data_stamp[r_begin, r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark  # shape = bs * seq_len * num_of_variates

    def __len__(self):  # ???
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self,data):
        return self.scaler.inverse_transform(data)

