import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Golbal import *
from Config import *


# 'timestamp', 'temperature', 'current', 'currentTXPower', 'currentRXPower', 'currentMultiRXPower1',
#          'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4', 'currentMultiTXPower1',
#          'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4', 'anomaly'

class FileData:
    def __init__(self, folder_path, file_name):
        self.folder_path = folder_path
        self.file_name = file_name
        self.file_path = os.path.join(folder_path, file_name)
        self.sn = os.path.splitext(file_name)[0]
        # print(self.folder_path)
        # print(self.file_name, self.sn)
        self.df = self.read_data()
        # print(self.df.columns.tolist())
        # self.print_line_x(line_index=0)

    def read_data(self):
        df = pd.read_csv(self.file_path)
        df[RowIndex] = [x for x in range(len(df))]
        return df

    def get_df(self, with_temp_outlier_rows=False, with_12null_rows=False, with_column_map=True, with_ts_delta=False):
        # print('++++')
        cut_row_indexes = set()
        if not with_temp_outlier_rows:
            temp_outlier_rows = set(self.get_temp_outlier_row_list())
            cut_row_indexes.update(temp_outlier_rows)
        if not with_12null_rows:
            null_12_rows = self.get_temp_null_row_list()
            cut_row_indexes.update(null_12_rows)
        ret_df = self.get_data(with_column_map)
        # ret_df['sn'] = self.sn
        # ret_df['step'] = [x for x in range(1, len(self.df)+1)]
        ret_df[Label] = 1 if ret_df[NAnomaly].sum() > 0 else 0
        # ret_df[RowIndex] = [x for x in range(len(ret_df))]
        # ret_df = ret_df[ret_df[NAnomaly] == ret_df[Label]]
        # print(ret_df.columns.to_list())
        cut_df = ret_df[ret_df[RowIndex].isin(cut_row_indexes)]
        if cut_row_indexes:
            ret_df = ret_df[~ret_df[RowIndex].isin(cut_row_indexes)]
        if with_ts_delta:
            ret_df[NTimeStamp] = ret_df[NTimeStamp].diff()
            ret_df[NTimeStamp] = ret_df[NTimeStamp].fillna(NaDefault)
        ret_df = ret_df[[*ORIGIN_FEATURE_LIST, NAnomaly, Label]]
        cut_df = cut_df[[*ORIGIN_FEATURE_LIST, NAnomaly, Label]]
        # print('###', ret_df.columns.tolist())
        # cut_df = cut_df[[NTimeStamp, *FEATURE_LIST, TrainLabel]]
        ret_df = ret_df.astype(np.float32)  # 为了避免爆内存，损失精度。TODO
        return ret_df, cut_df

    def get_data(self, with_column_map=True):
        df = self.df
        if with_column_map:
            # print('##############')
            # print(ColumnNameMap)
            df = df.rename(mapper=ColumnNameMap, axis=1)
        return df

    def get_temp_outlier_row_list(self):
        temp_outlier_row_list = self.df[self.df[Temperature] == TempOutlier][RowIndex].to_list()
        return temp_outlier_row_list

    def get_temp_null_row_list(self):
        temp_outlier_row_list = self.df[self.df[Temperature] == NaDefault][RowIndex].to_list()
        return temp_outlier_row_list

    def get_sn(self):
        return self.sn

    def get_value_by(self, column=Current, key='max'):
        if key == 'min':
            return self.df[column].min()
        elif key == 'max':
            return self.df[column].max()
        elif key == 'range':
            return self.df[column].max() - self.df[column].min()
        elif key == 'avg':
            return self.df[column].mean()  # avg、std、skew、kurt、len
        elif key == 'std':
            return self.df[column].std()
        elif key == 'skew':
            return self.df[column].skew()
        elif key == 'kurt':
            return self.df[column].kurt()
        elif key == 'len':
            return self.df[column].count()
        else:
            raise KeyError('The para key is not include!')

    def print_line_x(self, line_index):
        print('######### line:', line_index)
        for i, c in enumerate(self.df.columns):
            print(f'#{i}', c, '=', self.df[c][line_index])

    def pre_process(self, df):
        trainset1_df = df
        # Convert the timestamp column to datetime
        trainset1_df['timestamp'] = pd.to_datetime(trainset1_df['timestamp'])
        # Extract datetime features
        trainset1_df['year'] = trainset1_df['timestamp'].dt.year
        trainset1_df['month'] = trainset1_df['timestamp'].dt.month
        trainset1_df['day'] = trainset1_df['timestamp'].dt.day
        trainset1_df['hour'] = trainset1_df['timestamp'].dt.hour
        trainset1_df['minute'] = trainset1_df['timestamp'].dt.minute
        trainset1_df['dayofweek'] = trainset1_df['timestamp'].dt.dayofweek
        return df

    def plot(self):
        def get_label_num_str(feature_name):
            end_str = feature_name[-1]
            if end_str in {'1', '2', '3', '4'}:
                return end_str
            return '0'

        plt.figure(figsize=(5, 8))
        # 'timestamp', 'current', 'anomaly'
        numeric_features = ['currentRXPower', 'currentMultiRXPower1',
         'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4', ]
        plt.subplot(511)
        for feature in numeric_features:
            label = get_label_num_str(feature)
            plt.plot(self.df['timestamp'], self.df[feature], label=label)
        # plt.legend(bbox_to_anchor=(1, 2), loc=3)
        plt.legend()
        plt.title(f'{self.sn}_RX')

        numeric_features = [ 'currentTXPower', 'currentMultiTXPower1',
         'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4', ]
        plt.subplot(512)
        for feature in numeric_features:
            label = get_label_num_str(feature)
            plt.plot(self.df['timestamp'], self.df[feature], label=label)
        plt.legend()
        plt.title(f'{self.sn}_TX')

        plt.subplot(513)
        feature = 'temperature'
        plt.plot(self.df['timestamp'], self.df[feature], label=feature)
        # plt.legend(bbox_to_anchor=(1, 2), loc=3)
        plt.title(f'{self.sn}_Temp')

        plt.subplot(514)
        feature = 'current'
        plt.plot(self.df['timestamp'], self.df[feature], label=feature)
        plt.title(f'{self.sn}_Current')

        plt.subplot(515)
        feature = 'anomaly'
        plt.plot(self.df['timestamp'], self.df[feature], label=feature)
        plt.title(f'{self.sn}_Anomaly')
        plt.show()


if __name__ == "__main__":
    training1_path = r'C:\Users\z00381790\Desktop\光模块故障预测比赛\new_data\training'
    # training1_path = r'D:\WorkSpace\项目资料\20230801AI集群可靠性\光模块故障预测\光模块故障预测比赛202406\training1'
    # training1_path = r'D:\WorkSpace\项目资料\20230801AI集群可靠性\光模块故障预测\光模块故障预测比赛202406\training2'
    # training2_path = r'D:\Fault_prediction\optical_failure_prediction\2024_optical_failure_competition\training2\training2'
    # List to hold individual DataFrames

    # List all files in the directory
    for i, _filename in enumerate(os.listdir(training1_path)):
        if _filename not in {'000003826.csv'}:
            continue
        # Check if the file is a CSV
        if _filename.endswith('.csv'):
            data_obj = FileData(training1_path, _filename)
            data_obj.plot()
            if i >= 5:
                break
