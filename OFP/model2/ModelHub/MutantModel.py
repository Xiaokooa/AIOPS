from xgboost import XGBRFClassifier

from Config import *
from ModelHub.BaseModel import BaseModel
from MUTANT.Model import MUTANT

import torch
import numpy as np
import torch.optim as optim
from MUTANT.utils import get_data, get_data_dim, get_loader
from MUTANT.eval_method import bf_search
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


class ExpConfig:
    dataset = "MSL"  # "SMAP"
    val = 0.2  # 0.35  # the ratio of validation set, 在训练数据中切一些数据
    max_train_size = None  # `None` means full train set
    train_start = 0

    max_test_size = None  # `None` means full test set
    test_start = 0

    input_dim = 13  # TODO: 特征数
    batch_size = 120

    out_dim = 5   # the dimension of embedding
    window_length = 1  # TODO：20  ### ？？？
    hidden_size = 18  # 100  # the dimension of hidden layer in LSTM-based attention
    latent_size = 18  # 100  # the dimension of hidden layer in VAE # 必须小于input_dim*out_dim?
    N = 256


class MutantModel(BaseModel):
    MODEL_NAME = MnMutantModel

    def __init__(self):
        super().__init__()
        self.feature_list = list()
        config = ExpConfig()
        self.config = config
        w_size = config.input_dim * config.out_dim
        self.model_obj = MUTANT(config.input_dim, w_size, config.hidden_size, config.latent_size, config.batch_size, config.window_length, config.out_dim)
        self.optimizer = optim.Adam(self.model_obj.parameters(), lr=0.01)
        self.save_path = 'mutant_model.pt'
        self.percentile_threshold = 93
        self.scaler = MinMaxScaler()  # TODO：做好归一化

    def get_data(self, df):
        if np.any(sum(np.isnan(df)) != 0):
            print('Data contains null values. Will be replaced with 0')
            inx = np.isnan(df)
            df[inx] = 0
        if np.isinf(df).any():
            inx = np.isinf(df)
            print('Data contains inf values. Will be replaced with 100')
            df[inx] = 100
        return df

    def fit(self, train_data, retrain=True):
        # TODO: train_data 无需剔除异常，在这个函数中处理
        print('# Training Model...')
        config = self.config
        # train_x = train_data[self.feature_list]
        # train_y = train_data[TrainLabel]  # TODO：尝试不同标签

        # (train_data, _), (test_data, test_label) = \
        #     get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
        #              test_start=config.test_start)

        n = int(train_data.shape[0] * config.val)
        print('train_data columns:', train_data.columns.tolist())
        train_data = train_data[:-n]
        # print(train_data.columns.tolist())
        # print(train_data.head())
        # print(train_data[NAnomaly].min())
        # train_data = train_data[train_data[NAnomaly] == 0]  # 注意：训练数据需要剔除异常
        # self.percentile_threshold = int(100*len(train_data[train_data[NAnomaly] == 0])/len(train_data))
        self.percentile_threshold = 100*len(train_data[train_data[NAnomaly] == 0])/len(train_data)
        self.percentile_threshold = 100 - 0.9*(100 - self.percentile_threshold) # TODO
        train_data = train_data[train_data[Label] == 0]  # 注意：训练数据需要剔除异常
        train_data = train_data[[*ORIGIN_FEATURE_LIST, NAnomaly]]
        self.scaler.fit(self.get_data(np.asarray(train_data[self.feature_list], dtype=np.float32)))
        if not retrain:
            return
        # TODO：归一化？
        # test_label = test_label[:-n]

        val_data = train_data[-n:]
        val_label = val_data[NAnomaly]
        # Cut
        train_data = train_data[self.feature_list]
        val_data = val_data[self.feature_list]
        # 转出array
        train_data = np.asarray(train_data, dtype=np.float32)
        val_data = np.asarray(val_data, dtype=np.float32)
        train_data = self.get_data(train_data)
        val_data = self.get_data(val_data)
        val_label = np.asarray(val_label)
        print("train_data:", train_data.shape)
        print("val_data:", val_data.shape)
        # 归一化
        # self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        val_data = self.scaler.transform(val_data)

        train_data = train_data[
            np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]
        val_data = val_data[
            np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
        # test_data = test_data[
        #     np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]

        num_val = int(val_data.shape[0] / config.batch_size)
        con_val = val_data.shape[0] % config.batch_size
        # num_t = int(test_data.shape[0] / config.batch_size)
        # con_t = test_data.shape[0] % config.batch_size

        w_size = config.input_dim * config.out_dim

        train_loader = get_loader(train_data, batch_size=config.batch_size,
                                  window_length=config.window_length, input_size=config.input_dim, shuffle=True)
        val_loader = get_loader(val_data, batch_size=config.batch_size,
                                window_length=config.window_length, input_size=config.input_dim, shuffle=True)
        # test_loader = get_loader(test_data, batch_size=config.batch_size,
        #                          window_length=config.window_length, input_size=config.input_dim, shuffle=False)

        model = MUTANT(config.input_dim, w_size, config.hidden_size, config.latent_size, config.batch_size,
                       config.window_length, config.out_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        flag = 0
        f1 = -1
        for epoch in range(1):  # TODO: default: 10
            l = 0
            i = 0
            print('epoch:', epoch)
            for inputs in tqdm(train_loader):
                loss = model(inputs)
                loss.backward()
                if i % config.N == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                i += 1
            if flag == 1:
                model.load_state_dict(torch.load(self.save_path))
            val_score = model.is_anomaly(val_loader, num_val, con_val)
            t, th = bf_search(val_score, val_label[-len(val_score):], step_num=700)
            if t[0] > f1:
                f1 = t[0]
                torch.torch.save(model.state_dict(), self.save_path)
                flag = 1

    def set_features(self, feature_list: list):
        assert isinstance(feature_list, list)
        feature_set = set(feature_list)
        assert len(feature_set) == len(feature_list)
        self.feature_list = feature_list

    def predict(self, test_data, ret_proba=RETURN_EXTRA):
        print(f'# Running Model Predict...')
        config = self.config
        # 预测特征表格每一行结果即可, 输出结果至少包括三列，一列ts，一列predict，一列文件名
        if len(self.feature_list) == 0:
            raise RuntimeError('Please Set Feature Before Predict!!!')
        # test_data = test_data[:116]  # TODO: 截取100
        ret_data = test_data
        print('Feature Count:', len(self.feature_list))
        print('Feature Setting:', self.feature_list)
        test_data = test_data[[*self.feature_list]]
        # test_x = test_data[self.feature_list]
        # (train_data, _), (test_data, test_label) = \
        #     get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
        #              test_start=config.test_start)

        # n = int(train_data.shape[0] * config.val)
        # train_data = train_data[:-n]
        # train_data = train_data[train_data[NAnomaly == 0]]  # 注意：训练数据需要剔除异常
        # TODO：归一化？
        # test_label = test_label[:-n]

        # val_data = train_data[-n:]
        # val_label = val_data[NAnomaly]
        # 转出array
        test_data = np.asarray(test_data, dtype=np.float32)
        test_data = self.get_data(test_data)
        test_data = self.scaler.transform(test_data)
        # val_data = np.asarray(val_data)
        # val_label = np.asarray(val_label)
        print("test_data:", test_data.shape)
        # print("val_data:", val_data.shape)

        # train_data = train_data[
        #     np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]
        # val_data = val_data[
        #     np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
        test_data = test_data[
            np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]

        # num_val = int(val_data.shape[0] / config.batch_size)
        # con_val = val_data.shape[0] % config.batch_size
        num_t = int(test_data.shape[0] / config.batch_size)
        con_t = test_data.shape[0] % config.batch_size

        # w_size = config.input_dim * config.out_dim

        # train_loader = get_loader(train_data, batch_size=config.batch_size,
        #                           window_length=config.window_length, input_size=config.input_dim, shuffle=True)
        # val_loader = get_loader(val_data, batch_size=config.batch_size,
        #                         window_length=config.window_length, input_size=config.input_dim, shuffle=True)
        test_loader = get_loader(test_data, batch_size=config.batch_size,
                                 window_length=config.window_length, input_size=config.input_dim, shuffle=False)

        # test_data[Predict] = self.model_obj.predict(test_x)
        self.model_obj.load_state_dict(torch.load(self.save_path))
        start_time = datetime.now()
        print('start_time:', start_time)
        test_score = self.model_obj.is_anomaly(test_loader, num_t, con_t)
        end_time = datetime.now()
        print('end:', end_time)
        print('cost:', end_time - start_time)
        print('test_score:', min(test_score), max(test_score), np.median(test_score))
        score = np.asarray(test_score)
        thresh = np.percentile(score, float(self.percentile_threshold))
        # if pred is None:
        print(score.shape)
        predict = []
        for i in range(score.shape[0]):
            if score[i] > thresh:
                predict.append(1)
            else:
                predict.append(0)
        print('ret_data:', len(ret_data), 'predict:', len(predict))
        # ret_data = ret_data[:len(predict)]  # TODO
        ret_data = ret_data[config.window_length:]
        ret_data[Predict] = predict
        ret_cols = [TimeStamp, Predict, FILE_NAME]
        # if ret_proba:
        #     test_data[Proba] = self.model_obj.predict_proba(test_x)[:, 1]
        #     ret_cols.append(Proba)
        ret_data[TimeStamp] = ret_data[NTimeStamp]
        return ret_data[ret_cols]















