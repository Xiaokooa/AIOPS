from tqdm import tqdm

from Config import *
from ModelHub.BaseModel import BaseModel


class ExtraModel(BaseModel):
    MODEL_NAME = MnExtraModel

    def __init__(self):
        super().__init__()

    def fit(self, train_data, retrain=False):
        pass

    def predict(self, test_data):
        print(f'# Running Extra Predict...')
        # 预测特征表格每一行结果即可, 输出结果至少包括三列，一列ts，一列predict，一列文件名
        # print(test_data.columns.tolist())
        # print(test_data.head())
        cols = test_data.columns.tolist()
        if FeTempMin in cols:
            test_data[Predict] = test_data[FeTempMin].apply(lambda x: 1 if x == TempOutlier else 0)
        else:
            test_data[Predict] = test_data[NTemperature].apply(lambda x: 1 if x == TempOutlier else 0)
        ret_cols = [TimeStamp, Predict, FILE_NAME]
        test_data[TimeStamp] = test_data[NTimeStamp]
        return test_data[ret_cols]

