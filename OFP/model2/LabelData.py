import os

import pandas as pd

from Golbal import *


class LabelData:
    def __init__(self, folder_path, file_name, label_column, only_label_col=True):
        self.folder_path = folder_path
        self.file_name = file_name
        self.label_column = label_column
        self.file_path = os.path.join(folder_path, file_name)
        self.sn = os.path.splitext(file_name)[0]
        # print(self.folder_path)
        # print(self.file_name, self.sn)
        self.df = self.read_data(only_label_col)
        # print(self.df.columns.tolist())
        # self.print_line_x(line_index=0)
        self.label = self.get_label()
        self.first_label_ts = self.get_first_label_timestamp()

    def read_data(self, only_label_col):
        if only_label_col:
            df = pd.read_csv(self.file_path, usecols=[TimeStamp, self.label_column])
        else:
            df = pd.read_csv(self.file_path)
        df[RowIndex] = [x for x in range(len(df))]
        return df

    def get_label_and_first_ts(self):
        return self.label, self.first_label_ts

    def get_label(self):
        s = self.df[self.label_column].sum()
        if s > 0:
            return 1
        return 0

    def get_first_label_timestamp(self):
        if self.label == 0:
            return NoneLabelTsDefault
        return self.df[self.df[self.label_column] > 0][TimeStamp].min()

    def get_sn(self):
        return self.sn

    def get_df(self):
        return self.df

    def get_null_row_map_count(self):
        null_count_df = self.df.isnull().sum(axis=1)
        # null_count_df = self.df.isna().sum(axis=1)
        # print(null_count_df.head())
        # print(self.df.columns.tolist())
        # print(self.df['currentMultiTXPower4'][415:418])
        null_row_map_count = {k: v for k, v in null_count_df.items() if v > 0}
        return null_row_map_count

    def get_temp_outlier_row_list(self):
        temp_outlier_row_list = self.df[self.df[Temperature] == TempOutlier].index.to_list()
        return temp_outlier_row_list


