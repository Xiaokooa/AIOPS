import os

import pandas as pd
from tqdm import tqdm

from FeatureExtractor import FeatureExtractor
from Golbal import *
from Config import *
import numpy as np


class FeatureData:
    def __init__(self, folder_path, file_name, label_column=NAnomaly, only_feature_col=False):
        self.folder_path = folder_path
        self.file_name = file_name
        self.label_column = label_column
        self.file_path = os.path.join(folder_path, file_name)
        self.sn = os.path.splitext(file_name)[0]
        # print(self.folder_path)
        # print(self.file_name, self.sn)
        self.df = self.read_data(only_feature_col)
        # print(self.df.columns.tolist())
        # self.print_line_x(line_index=0)
        # self.label = self.get_label()
        # self.first_label_ts = self.get_first_label_timestamp()

    def read_data(self, only_feature_col):
        if only_feature_col:
            feature_cols = FeatureExtractor.generate_feature_names()
            df = pd.read_csv(self.file_path, usecols=[TimeStamp, *feature_cols])
        else:
            df = pd.read_csv(self.file_path)
        df[RowIndex] = [x for x in range(len(df))]
        return df

    def get_sn(self):
        return self.sn

    def get_df(self, with_temp_outlier_rows=False, with_12null_rows=False, with_ts_delta=True):
        cut_row_indexes = set()
        if not with_temp_outlier_rows:
            temp_outlier_rows = set(self.get_temp_outlier_row_list())
            cut_row_indexes.update(temp_outlier_rows)
        if not with_12null_rows:
            null_12_rows = self.get_temp_null_row_list()
            cut_row_indexes.update(null_12_rows)
        ret_df = self.df
        # ret_df = ret_df[ret_df[NAnomaly] == ret_df[Label]]
        # print(ret_df.columns.to_list())
        cut_df = ret_df[ret_df[RowIndex].isin(cut_row_indexes)]
        if cut_row_indexes:
            ret_df = ret_df[~ret_df[RowIndex].isin(cut_row_indexes)]
        # 将数据中的-999处理一下，主要是Skew
        feature_cols = FeatureExtractor.generate_feature_names(by_default=True)
        for c in feature_cols:
            if c.endswith(MeSkew):
                ret_df[c].replace(NaDefault, 0, inplace=True)
        if with_ts_delta:
            ret_df[NTimeStampDelta] = ret_df[NTimeStamp].diff()
            ret_df[NTimeStampDelta] = ret_df[NTimeStampDelta].fillna(NaDefault)
            cut_df[NTimeStampDelta] = cut_df[NTimeStamp].diff()
            cut_df[NTimeStampDelta] = cut_df[NTimeStampDelta].fillna(NaDefault)
            # print('%%%')
        # print(ret_df.columns.tolist())
        # ret_df = ret_df[[NTimeStamp, *DEFAULT_FEATURE_LIST, TrainLabel]]
        cols = ret_df.columns.tolist()
        if MODEL_NAME == MnTreeModel:
            feature_list = DEFAULT_FEATURE_LIST
        else:
            feature_list = [NTimeStamp, *FEATURE_LIST]
        if TrainLabel in cols:
            ret_df = ret_df[[*feature_list, TrainLabel]]   # TODO： 换模型需要修改
        else:
            ret_df = ret_df[[*feature_list]]
        # cut_df = cut_df[[NTimeStamp, *DEFAULT_FEATURE_LIST, TrainLabel]]
        cols = cut_df.columns.tolist()
        if TrainLabel in cols:
            cut_df = cut_df[[*feature_list, TrainLabel]]
        else:
            cut_df = cut_df[[*feature_list]]
        ret_df = ret_df.astype(np.float32)  # 为了避免爆内存，损失精度。TODO
        return ret_df, cut_df

    # def get_null_row_map_count(self):
    #     # 注意：特征文件中的空值已经是-999
    #     null_count_df = self.df.isnull().sum(axis=1)
    #     # null_count_df = self.df.isna().sum(axis=1)
    #     # print(null_count_df.head())
    #     # print(self.df.columns.tolist())
    #     # print(self.df['currentMultiTXPower4'][415:418])
    #     null_row_map_count = {k: v for k, v in null_count_df.items() if v > 0}
    #     return null_row_map_count

    def get_temp_outlier_row_list(self):
        temp_outlier_row_list = self.df[self.df[NTemperature] == TempOutlier][RowIndex].to_list()
        return temp_outlier_row_list

    def get_temp_null_row_list(self):
        temp_outlier_row_list = self.df[self.df[NTemperature] == NaDefault][RowIndex].to_list()
        return temp_outlier_row_list
