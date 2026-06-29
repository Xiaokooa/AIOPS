from tqdm import tqdm

from Config import *
from ModelHub.BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import pickle
import time


class TreeModel(BaseModel):
    MODEL_NAME = MnTreeModel

    def __init__(self):
        super().__init__()
        self.feature_list = list()
        self.model_obj = RandomForestClassifier(n_estimators=100, random_state=2024, verbose=3)  # n_estimators表示树的数量
        # self.model_obj = XGBClassifier(n_estimators=10, random_state=2024)  # n_estimators表示树的数量
        # self.model_obj = XGBRFClassifier(n_estimators=10, random_state=2024)  # n_estimators表示树的数量
        # self.model_obj = LGBMClassifier(n_estimators=10, random_state=2024)  # n_estimators表示树的数量

    def fit(self, train_data, retrain=True):
        print('# Training Model...')
        train_x = train_data[self.feature_list]
        train_y = train_data[TrainLabel]  # TODO：尝试不同标签
        self.model_obj = self.model_obj.fit(train_x, train_y)
        ts = int(time.time())
        file_path = f'rf_{ts}.pkl'
        with open(file_path, 'wb') as _f:
            pickle.dump(self.model_obj, _f)

    def set_features(self, feature_list: list):
        assert isinstance(feature_list, list)
        feature_set = set(feature_list)
        assert len(feature_set) == len(feature_list)
        self.feature_list = feature_list

    def predict(self, test_data, ret_proba=RETURN_EXTRA):
        print(f'# Running Model Predict...')
        # 预测特征表格每一行结果即可, 输出结果至少包括三列，一列ts，一列predict，一列文件名
        if len(self.feature_list) == 0:
            raise RuntimeError('Please Set Feature Before Predict!!!')
        print('Feature Count:', len(self.feature_list))
        print('Feature Setting:', self.feature_list)
        test_x = test_data[self.feature_list]
        # test_data_cols = set(test_data.columns.tolist())
        # print('Match:', self.rule_set - test_data_cols)
        # test_data[RuleMatchCnt] = 0
        # for _rule in RULE_SET:
        #     test_data[RuleMatchCnt] += test_data[_rule]
        # test_data[Predict] = self.model_obj.predict(test_x)
        ret_cols = [TimeStamp, Predict, FILE_NAME]
        if ret_proba:
            test_data[Proba] = self.model_obj.predict_proba(test_x)[:, 1]
            test_data[Predict] = test_data[Proba].apply(lambda x: 1 if x >= 0.3 else 0)  # TODO
            ret_cols.append(Proba)
        print(test_data.columns.tolist())
        # cols = test_data.columns.tolist()
        # if NTimeStamp in cols:
        #     test_data[TimeStamp] = test_data[NTimeStamp]
        # else:
        #     assert TimeStamp in cols
        test_data[TimeStamp] = test_data[NTimeStamp]
        return test_data[ret_cols]


