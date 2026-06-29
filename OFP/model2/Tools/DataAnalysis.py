import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from FileData import FileData


class DataAnalysis:
    def __init__(self, pos_folder, neg_folder):
        self.pos_folder = pos_folder
        self.neg_folder = neg_folder
        self.pos_file_list = os.listdir(pos_folder)
        self.neg_file_list = os.listdir(neg_folder)

    def list_file_path(self, folder_path):
        file_list = os.listdir(folder_path)
        return [os.path.join(folder_path, x) for x in file_list]

    def plot_1d_data_distribution(self, dim, value_type, fig_type='default'):

        def get_one_folder_data(folder, file_list, _dim, _value_type):
            _dim_values = []
            for _file_name in tqdm(file_list):
                file_data = FileData(folder, file_name=_file_name)
                value_1 = file_data.get_value_by(column=_dim, key=value_type)
                _dim_values.append(value_1)
            return _dim_values

        print(dim, value_type)
        label_name = 'label'
        folder1_dim_1_values = get_one_folder_data(self.pos_folder, self.pos_file_list, dim, value_type)
        folder2_dim_1_values = get_one_folder_data(self.neg_folder, self.neg_file_list, dim, value_type)
        all_dim_1_values = [*folder2_dim_1_values, *folder1_dim_1_values]
        all_folder_labels = [0]*len(folder2_dim_1_values) + [1]*len(folder1_dim_1_values)
        df_2d = pd.DataFrame(data={dim: all_dim_1_values, label_name: all_folder_labels})
        plt.figure(figsize=(8, 8))
        # plt.subplot(221)
        if fig_type == 'violin':
            sns.set_theme(style="dark")
            sns.violinplot(data=df_2d, x=label_name, y=dim, hue=label_name,
                           split=True, inner="quart",  # fill=False,
                           )
        else:
            sns.displot(x=dim, hue=label_name, data=df_2d)
            plt.title(f'{dim}_{value_type}')
        # df_2d = df_2d.sort_values(by=label_name, ascending=False)
        # plt.subplot(222)
        # sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d)
        # plt.title(f'{dim_1}&{dim_2}_{value_type}')
        # plt.subplot(223)
        # sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d[df_2d[label_name] == 0])
        # plt.title(f'{dim_1}&{dim_2}_{value_type}_neg')
        # plt.subplot(224)
        # sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d[df_2d[label_name] == 1])
        # plt.title(f'{dim_1}&{dim_2}_{value_type}_pos')
        plt.show()

    def plot_2d_data_distribution(self, dim_1, dim_2, value_type, fig_type='default', out_folder=None):

        def get_one_folder_data(folder, file_list, _dim_1, _dim_2, _value_type):
            _dim_1_values, _dim_2_values = [], []
            for _file_name in tqdm(file_list):
                file_data = FileData(folder, file_name=_file_name)
                value_1 = file_data.get_value_by(column=_dim_1, key=value_type)
                value_2 = file_data.get_value_by(column=_dim_2, key=value_type)
                _dim_1_values.append(value_1)
                _dim_2_values.append(value_2)
            return _dim_1_values, _dim_2_values

        # print()
        print(f"{dim_1}&{dim_2} # {value_type}")
        # print()
        label_name = 'label'
        folder1_dim_1_values, folder1_dim_2_values = get_one_folder_data(self.pos_folder, self.pos_file_list, dim_1, dim_2, value_type)
        folder2_dim_1_values, folder2_dim_2_values = get_one_folder_data(self.neg_folder, self.neg_file_list, dim_1, dim_2, value_type)
        all_dim_1_values = [*folder2_dim_1_values, *folder1_dim_1_values]
        all_dim_2_values = [*folder2_dim_2_values, *folder1_dim_2_values]
        all_folder_labels = [0]*len(folder2_dim_1_values) + [1]*len(folder1_dim_1_values)
        df_2d = pd.DataFrame(data={dim_1: all_dim_1_values, dim_2: all_dim_2_values, label_name: all_folder_labels})
        if fig_type == 'pair':
            # sns.pairplot(df_2d, hue=label_name, markers=['o', 's'])
            sns.pairplot(df_2d, hue=label_name)
            # plt.show()
        elif fig_type == 'joint':
            sns.jointplot(data=df_2d, x=dim_1, y=dim_2, hue=label_name)
        elif fig_type == 'rel':
            sns.relplot(data=df_2d, x=dim_1, y=dim_2, hue=label_name)
        else:
            # category_colors = {1: "orange", 0: "green"}
            category_colors = {1: "lightcoral", 0: "lightgreen"}  # 这个还行
            # category_colors = {1: "orangered", 0: "springgreen"}
            # category_colors = {1: "darkred", 0: "deepskyblue"}
            # category_colors = {1: "darkred", 0: "darkgreen"}
            plt.figure(figsize=(8, 8))
            # plt.figure()
            plt.subplot(221)
            sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d, palette=category_colors)
            plt.title(f'{dim_1} & {dim_2}')
            df_2d = df_2d.sort_values(by=label_name, ascending=False)
            plt.subplot(222)
            sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d, palette=category_colors)
            plt.title(f'{value_type}')
            plt.subplot(223)
            sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d[df_2d[label_name] == 0], palette=category_colors)
            # plt.title(f'only_neg')
            plt.subplot(224)
            sns.scatterplot(x=dim_1, y=dim_2, hue=label_name, data=df_2d[df_2d[label_name] == 1], palette=category_colors)
            # plt.title(f'only_pos')
        plt.tight_layout()
        if out_folder is None:
            plt.show()
        else:
            out_fig_path = os.path.join(out_folder, f'{dim_1}&{dim_2}_{value_type}.png')
            plt.savefig(out_fig_path)
            print('write:', out_fig_path)


if __name__ == '__main__':
    # 'timestamp', 'temperature', 'current', 'currentTXPower', 'currentRXPower', 'currentMultiRXPower1',
    #          'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4', 'currentMultiTXPower1',
    #          'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4', 'anomaly'
    # folder_path1 = r'D:\WorkSpace\项目资料\20230801AI集群可靠性\光模块故障预测\光模块故障预测比赛202406\training1'
    # folder_path2 = r'D:\WorkSpace\项目资料\20230801AI集群可靠性\光模块故障预测\光模块故障预测比赛202406\training2'
    pos_folder = r'C:\Users\z00381790\Desktop\光模块故障预测比赛\new_data\training1'
    neg_folder = r'C:\Users\z00381790\Desktop\光模块故障预测比赛\new_data\training2'
    data_analyzer = DataAnalysis(pos_folder, neg_folder)
    # min, max, range、avg、std、skew、kurt、len
    # value_type_list = ['min', 'max', 'range', 'avg', 'std', 'skew', 'kurt', 'len']
    # data_analyzer.plot_2d_data_distribution(dim_1='temperature', dim_2='current', value_type='avg')

    # TODO:
    fig_folder = r'C:\Users\z00381790\Desktop\光模块故障预测比赛\new_data\fig'
    value_type_list = ['min', 'max', 'range', 'std', 'skew', 'kurt']
    # _dim_1, _dim_2 = 'temperature', 'current'
    # _dim_1, _dim_2 = 'currentTXPower', 'currentRXPower'
    # print('*'*36)
    # _dim_1, _dim_2 = 'currentMultiRXPower1', 'currentMultiRXPower2'
    for _dimm_pair in [('temperature', 'current'),
                       ('currentTXPower', 'currentRXPower'),
                       ('currentMultiRXPower1', 'currentMultiRXPower2'),
                       ('currentMultiRXPower3', 'currentMultiRXPower4'),
                       ('currentMultiTXPower1', 'currentMultiTXPower2'),
                       ('currentMultiTXPower3', 'currentMultiTXPower4')]:
        _dim_1, _dim_2 = _dimm_pair
        for _value_type in value_type_list:
            data_analyzer.plot_2d_data_distribution(_dim_1, _dim_2, _value_type, fig_type='default', out_folder=fig_folder)
    # print('*' * 36)
    # _dim_1, _dim_2 = 'currentMultiRXPower3', 'currentMultiRXPower4'
    # for _value_type in value_type_list:
    #     data_analyzer.plot_2d_data_distribution(_dim_1, _dim_2, _value_type, fig_type='default')
    # print('*' * 36)
    # _dim_1, _dim_2 = 'currentMultiTXPower1', 'currentMultiTXPower2'
    # for _value_type in value_type_list:
    #     data_analyzer.plot_2d_data_distribution(_dim_1, _dim_2, _value_type, fig_type='default')
    # print('*' * 36)
    # _dim_1, _dim_2 = 'currentMultiTXPower3', 'currentMultiTXPower4'
    # for _value_type in value_type_list:
    #     data_analyzer.plot_2d_data_distribution(_dim_1, _dim_2, _value_type, fig_type='default')

    # data_analyzer.plot_2d_data_distribution(dim_1='temperature', dim_2='current', value_type='min', fig_type='default')
    # data_analyzer.plot_2d_data_distribution(dim_1='temperature', dim_2='current', value_type='min', fig_type='pair')
    # data_analyzer.plot_2d_data_distribution(dim_1='temperature', dim_2='current', value_type='min', fig_type='joint')
    # data_analyzer.plot_2d_data_distribution(dim_1='temperature', dim_2='current', value_type='min', fig_type='rel')

    # data_analyzer.plot_1d_data_distribution(dim='current', value_type='min', fig_type='violin')
    # data_analyzer.plot_1d_data_distribution(dim='current', value_type='min', fig_type='default')  # displot




# 'timestamp', 'temperature', 'current', 'currentTXPower', 'currentRXPower', 'currentMultiRXPower1',
#          'currentMultiRXPower2', 'currentMultiRXPower3', 'currentMultiRXPower4', 'currentMultiTXPower1',
#          'currentMultiTXPower2', 'currentMultiTXPower3', 'currentMultiTXPower4', 'anomaly'