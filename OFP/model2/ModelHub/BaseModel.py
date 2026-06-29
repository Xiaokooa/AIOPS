from Golbal import *


class BaseModel:
    MODEL_NAME = MnBaseModel

    def __init__(self):
        self.model_obj = None
        print(f'***** Current Model: {self.MODEL_NAME} *****')

    def get_model_name(self):
        return self.MODEL_NAME

    def fit(self, train_data, retrain=True):
        pass

    def predict(self, test_data):
        # 预测特征表格每一行结果即可, 输出结果至少包括三列，一列ts，一列predict，一列文件名
        pass

    def save_model(self):
        # 对于要训练比较久的模型，最好保留一下中间结果。
        pass
