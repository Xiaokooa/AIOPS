import random

import pandas as pd

from Config import *
from Golbal import *


def split_samples_to_train_test(pos_csv_list, neg_csv_list, out_cv_file_path, cross_folder_count: int = 5):
    """

    :param pos_csv_list:
    :param neg_csv_list:
    :param cross_folder_count: 默认交叉验证折数5
    :return:
    """
    def get_cross_folder_indexes(all_cnt, folder_cnt):
        each_folder_obj_cnt = int(all_cnt/folder_cnt)
        indexes = []
        for i in range(1, folder_cnt+1):
            cur_folder_obj_cnt = each_folder_obj_cnt
            if i == folder_cnt:
                cur_folder_obj_cnt = all_cnt - (folder_cnt-1)*each_folder_obj_cnt
            indexes.extend([i] * cur_folder_obj_cnt)
        return indexes

    all_pos_cnt = len(pos_csv_list)
    pos_indexes = get_cross_folder_indexes(all_pos_cnt, folder_cnt=cross_folder_count)
    pos_indexes = random.sample(pos_indexes, k=all_pos_cnt)  # 打乱顺序
    all_neg_cnt = len(neg_csv_list)
    neg_indexes = get_cross_folder_indexes(all_neg_cnt, folder_cnt=cross_folder_count)
    neg_indexes = random.sample(neg_indexes, k=all_neg_cnt)
    cv_df = pd.DataFrame(data={
        FILE_NAME: [*pos_csv_list, *neg_csv_list],  # 文件名
        FOLDER_INDEX: [*pos_indexes, *neg_indexes],  # 折号的索引，第几折
        Label: [1]*len(pos_indexes) + [0]*len(neg_indexes),  # 记录正负
    })
    cv_df.to_csv(out_cv_file_path, index=False)


if __name__ == '__main__':
    split_samples_to_train_test(POS_CSV_LIST, NEG_CSV_LIST, TRAIN_TEST_SET_INDEX_CSV, cross_folder_count=3)
