import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from Golbal import *
from Config import *

from FileData import FileData
from LabelData import LabelData


class DataCleaner:
    def __init__(self, label_folder):
        self.label_folder = label_folder
        self.label_file_list = os.listdir(label_folder)

    def list_file_path(self, folder_path):
        file_list = os.listdir(folder_path)
        return [os.path.join(folder_path, x) for x in file_list]

    def analyze_null_row_of_feature(self):
        print('# Analyze Null Row')
        for origin_file_name in tqdm(self.label_file_list):
            label_data_obj = LabelData(self.label_folder, origin_file_name, label_column=Anomaly, only_label_col=False)
            # print(origin_file_name)
            null_row_map_count = label_data_obj.get_null_row_map_count()
            # print(null_row_map_count)
            # break
            temp_outlier_rows = label_data_obj.get_temp_outlier_row_list()
            df = label_data_obj.get_df()
            temp_outlier_values = df[df[RowIndex].isin(temp_outlier_rows)][Anomaly].tolist()
            for v in temp_outlier_values:
                if v != 1:
                    raise RuntimeError('Found anomaly not 1 in temp -255 !')
            for null_row, count in null_row_map_count.items():
                if count not in {12, 6}:
                    print(origin_file_name, null_row, count)


if __name__ == '__main__':
    data_cleaner = DataCleaner(LABEL_FOLDER)
    data_cleaner.analyze_null_row_of_feature()


