import os
import shutil

import pandas as pd
from tqdm import tqdm

from Golbal import *

if __name__ == '__main__':
    origin_folder = r'C:\Users\光模块故障预测比赛\new_data\training'
    pos_folder = r'C:\Users\光模块故障预测比赛\new_data\training1'
    neg_folder = r'C:\Users\光模块故障预测比赛\new_data\training2'
    file_list = os.listdir(origin_folder)
    for f in tqdm(file_list):
        file_path = os.path.join(origin_folder, f)
        df = pd.read_csv(file_path)
        if df[Anomaly].sum() > 0:
            output_folder = pos_folder
        else:
            output_folder = neg_folder
        output_file_path = os.path.join(output_folder, f)
        try:
            # 使用shutil的copy()函数复制文件
            shutil.copy(file_path, output_file_path)
            # print("文件复制成功！")
        except FileNotFoundError:
            print("源文件不存在，请检查路径是否正确。")
        except PermissionError:
            print("没有足够的权限复制文件。")
        except Exception as e:
            print(f"复制文件时发生错误：{e}")


