import os
from tqdm import tqdm
import pandas as pd
from Golbal import *


class ResultMerger:
    def __init__(self):
        pass

    def sum_same_len_result_by_threshold(self, pred_ge_thresh_cnt, out_folder, *in_folders):
        filename_list = os.listdir(in_folders[0])
        for _file_name in tqdm(filename_list):
            for i, _folder in enumerate(in_folders):
                _file_path = os.path.join(_folder, _file_name)
                _path = _file_path
                # if not os.path.isfile(_path):
                #     print(f'Warning: {_path} is not a file!!!')
                #     continue
                if i == 0:
                    if not os.path.isfile(_path):
                        print(f'Warning: {_path} is not a file!!!')
                        continue
                    sum_df = pd.read_csv(_file_path)
                else:
                    _file_path = os.path.join(_folder, f'feat_{_file_name}')  # TODO
                    if not os.path.isfile(_file_path):
                        print(f'Warning: {_file_path} is not a file!!!')
                        continue
                    add_df = pd.read_csv(_file_path)
                    # assert len(sum_df) == len(add_df)
                    sum_df[Predict] = sum_df[Predict] + add_df[Predict]
                    if Proba in sum_df.columns.tolist() and Proba in add_df.columns.tolist():
                        sum_df[Proba] = sum_df[Proba] + add_df[Proba]
            sum_df[Predict] = sum_df[Predict].apply(lambda x: 1 if x >= pred_ge_thresh_cnt else 0)
            if Proba in sum_df.columns.tolist():
                sum_df[Proba] = sum_df[Proba]/len(in_folders)
            out_file_path = os.path.join(out_folder, _file_name)
            # print(out_file_path)
            sum_df.to_csv(out_file_path, index=False)

    def merge_rule_with_another(self, rule_folder, another_folder, out_folder):
        filename_list = os.listdir(rule_folder)
        add_cnt_sum = 0
        for _file_name in tqdm(filename_list):
            _file_path = os.path.join(rule_folder, _file_name)
            _path = _file_path
            if not os.path.isfile(_path):
                print(f'Warning: {_path} is not a file!!!')
                continue
            rule_df = pd.read_csv(_file_path)
            # rule_df_len = len(rule_df)
            _file_path = os.path.join(another_folder, _file_name)
            another_df = pd.read_csv(_file_path)
            # another_df_len = len(another_df)
            pred_pos_ts_set = set(another_df[another_df[Predict] > 0][TimeStamp])
            old_pred_count = rule_df[Predict].sum()
            rule_df[Predict] = list(map(lambda x, y: 1 if x in pred_pos_ts_set else y, rule_df[TimeStamp], rule_df[Predict]))
            new_pred_count = rule_df[Predict].sum()
            add_cnt_sum += new_pred_count - old_pred_count
            out_file_path = os.path.join(out_folder, _file_name)
            rule_df.to_csv(out_file_path, index=False)

    def compare_result(self, folder1, folder2):
        filename_list = os.listdir(folder1)
        for _file_name in tqdm(filename_list):
            _file_path = os.path.join(folder1, _file_name)
            df1 = pd.read_csv(_file_path)
            _file_path = os.path.join(folder1, _file_name)
            df2 = pd.read_csv(_file_path)
            assert len(df1) == len(df2)

if __name__ == "__main__":
    # folder1 = r"D:\new_data\predict_result\tree_predict_default"
    # folder2 = r"D:\new_data\predict_result\folder_cross_split_qiaoyu"
    # folder3 = r"D:\new_data\predict_result\rule_predict"
    # out_folder = r'D:\new_data\predict_result\tree_predict_merge_qiaoyu'
    # ResultMerger().sum_same_len_result_by_threshold(1, out_folder, folder1, folder2)

    # folder1 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\folder_1"
    # folder2 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\folder_2"
    # folder3 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\folder_1"
    # out_folder = r'D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\tree_result'
    # ResultMerger().sum_same_len_result_by_threshold(1, out_folder, folder1, folder2, folder3)

    # folder2 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\tree_result"
    # folder1 = r"D:\new_data\predict_result\test_result_qiaoyu"
    # # folder3 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\folder_1"
    # out_folder = r'D:\new_data\predict_result\tree_predict_final_result'
    # ResultMerger().sum_same_len_result_by_threshold(1, out_folder, folder1, folder2)

    folder2 = r"D:\new_data\test"
    folder1 = r"D:\new_data\predict_result\tree_predict_final_result"
    # folder3 = r"D:\new_data\predict_result\tree_predict_default\test_threshold_0.5\folder_1"
    # out_folder = r'D:\new_data\predict_result\tree_predict_final_result'
    ResultMerger().compare_result(folder1, folder2)
