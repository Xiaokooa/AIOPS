import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from Golbal import *


# 训练集仅需要无标签的正常数据
# 考虑是否归一化
def convert_to_matrix(input_csv_folder, output_folder):
    print('# convert_to_matrix')
    file_list = os.listdir(input_csv_folder)
    for _file_name in tqdm(file_list):
        file_path = os.path.join(input_csv_folder, _file_name)
        attack_and_label = pd.read_csv(file_path)
        attack_and_label[TimeStamp] = attack_and_label[TimeStamp].diff()
        attack_and_label.rename(mapper=ColumnNameMap, inplace=True)  # 列名重映射
        print('attack_and_label', attack_and_label.shape)

        # attack = np.asarray(attack)




if __name__ == '__main__':
    pass
