import os.path

import pandas as pd
from tqdm import tqdm

from Config import *
from Golbal import *
from ModelHub.BaseModel import BaseModel
from ModelHub.RuleModel import RuleModel
from FileData import FileData
from FeatureData import FeatureData
from ModelHub.ExtraModel import ExtraModel


class ModelRunner:
    def __init__(self, model_type: BaseModel):
        self.model_class_obj = model_type
        self.model_name = self.model_class_obj.get_model_name()
        self.folder_map_file_names = self.__get_cross_folders()
        self.file_map_label = self.__get_file_map_label()

    def __train_model_in_one_folder(self, train_data, retrain=True):
        self.model_class_obj.fit(train_data, retrain)

    def __predict_model_in_one_folder(self, test_data, extra_df):
        not_null_df_predict_result = self.model_class_obj.predict(test_data)
        null_df_predict_result = ExtraModel().predict(extra_df)
        total_predict_result_df = pd.concat([not_null_df_predict_result, null_df_predict_result], axis=0)
        total_predict_result_df.sort_values(by=[FILE_NAME, TimeStamp], ascending=True, inplace=True)
        total_predict_result_df.reset_index(drop=True, inplace=True)
        return total_predict_result_df

    def train_predict_in_one_folder(self, train_data, test_data, extra_df, retrain=True):
        self.__train_model_in_one_folder(train_data, retrain)
        return self.__predict_model_in_one_folder(test_data, extra_df)

    @staticmethod
    def __get_cross_folders():
        cv_df = pd.read_csv(TRAIN_TEST_SET_INDEX_CSV)
        unique_folder_nums = cv_df[FOLDER_INDEX].unique().tolist()
        folder_map_file_names = dict()
        for _folder_num in unique_folder_nums:
            folder_map_file_names[_folder_num] = cv_df[cv_df[FOLDER_INDEX] == _folder_num][FILE_NAME].tolist()
        return folder_map_file_names

    @staticmethod
    def __get_file_map_label():
        cv_df = pd.read_csv(TRAIN_TEST_SET_INDEX_CSV)
        mapping_dict = cv_df.set_index(FILE_NAME)[Label].to_dict()
        return mapping_dict

    def __get_train_data_by_folder(self, folder_list):
        def contact_many_dataframes(df_list):
            # 假设df_list是一个包含大量DataFrame的列表
            batch_size = 100  # 根据实际情况调整批处理大小
            result_df = pd.DataFrame()  # 初始化结果DataFrame

            for i in range(0, len(df_list), batch_size):
                print('contact : ', i)
                batch_dfs = df_list[i:i + batch_size]
                temp_df = pd.concat(batch_dfs, ignore_index=True)  # ignore_index=True重置索引
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            return result_df

        all_train_files = []
        for _folder in folder_list:
            all_train_files.extend(self.folder_map_file_names[_folder])
        train_data_list = []
        print('# Reading Train Data...')
        train_df = pd.DataFrame()
        for _file_name in tqdm(all_train_files):
            _file_label = self.file_map_label[_file_name]
            if self.model_name in ORIGIN_FEATURE_MODELS:
                feature_file_obj = FileData(LABEL_FOLDER, _file_name)
            else:
                feature_file_name = f"{FEATURE_FILE_PREFIX}{_file_name}"  # 原始文件名和特征文件名不一样
                if _file_label > 0:
                    # file_path = os.path.join(FEATURE_POS_FOLDER, feature_file_name)
                    feature_file_obj = FeatureData(FEATURE_POS_FOLDER, feature_file_name)
                else:
                    # file_path = os.path.join(FEATURE_NEG_FOLDER, feature_file_name)
                    feature_file_obj = FeatureData(FEATURE_NEG_FOLDER, feature_file_name)
            # one_file_df = pd.read_csv(file_path)
            one_file_df, extra_data_df = feature_file_obj.get_df(with_temp_outlier_rows=False, with_12null_rows=False)
            train_data_list.append(one_file_df)
            # train_df = train_df.append(one_file_df, ignore_index=True)
            # train_df = pd.concat([train_df, one_file_df], ignore_index=True)
        train_df = pd.concat(train_data_list, axis=0)
        # train_df = contact_many_dataframes(train_data_list)
        return train_df

    def __get_test_data_by_folder(self, folder_list):
        all_test_files = []
        for _folder in folder_list:
            all_test_files.extend(self.folder_map_file_names[_folder])
        test_data_list = []
        extra_data_list = []
        print()
        print('# Reading Test Data...', len(all_test_files))
        for _file_name in tqdm(all_test_files):
            _file_label = self.file_map_label[_file_name]
            if self.model_name in ORIGIN_FEATURE_MODELS:
                feature_file_obj = FileData(LABEL_FOLDER, _file_name)
            else:
                feature_file_name = f"{FEATURE_FILE_PREFIX}{_file_name}"  # 原始文件名和特征文件名不一样
                if _file_label > 0:
                    # file_path = os.path.join(FEATURE_POS_FOLDER, feature_file_name)
                    feature_file_obj = FeatureData(FEATURE_POS_FOLDER, feature_file_name)
                else:
                    # file_path = os.path.join(FEATURE_NEG_FOLDER, feature_file_name)
                    feature_file_obj = FeatureData(FEATURE_NEG_FOLDER, feature_file_name)
            # one_file_df = pd.read_csv(file_path)
            # print('$$$$')
            one_file_df, extra_data_df = feature_file_obj.get_df(with_temp_outlier_rows=False, with_12null_rows=False)
            one_file_df[FILE_NAME] = _file_name  # 增加文件名关联，以用于分割预测结果。
            extra_data_df[FILE_NAME] = _file_name  # 增加文件名关联，以用于分割预测结果。
            test_data_list.append(one_file_df)
            extra_data_list.append(extra_data_df)
        test_df = pd.concat(test_data_list, axis=0)
        extra_df = pd.concat(extra_data_list, axis=0)
        return test_df, extra_df

    def __get_test_data_by_folder_path(self, folder_path):
        all_test_files = os.listdir(folder_path)
        test_data_list = []
        extra_data_list = []
        print('# Reading Test Data...')
        for _file_name in tqdm(all_test_files):
            # feature_file_name = f"{FEATURE_FILE_PREFIX}{_file_name}"  # 原始文件名和特征文件名不一样
            feature_file_name = f"{_file_name}"  # 原始文件名和特征文件名不一样
            feature_file_obj = FeatureData(FEATURE_TEST_FOLDER, feature_file_name)
            # _file_label = self.file_map_label[_file_name]
            # if self.model_name in ORIGIN_FEATURE_MODELS:
            #     feature_file_obj = FileData(LABEL_FOLDER, _file_name)
            # else:
            #     feature_file_name = f"{FEATURE_FILE_PREFIX}{_file_name}"  # 原始文件名和特征文件名不一样
            #     if _file_label > 0:
            #         # file_path = os.path.join(FEATURE_POS_FOLDER, feature_file_name)
            #         feature_file_obj = FeatureData(FEATURE_POS_FOLDER, feature_file_name)
            #     else:
            #         # file_path = os.path.join(FEATURE_NEG_FOLDER, feature_file_name)
            #         feature_file_obj = FeatureData(FEATURE_NEG_FOLDER, feature_file_name)
            # one_file_df = pd.read_csv(file_path)
            one_file_df, extra_data_df = feature_file_obj.get_df(with_temp_outlier_rows=False, with_12null_rows=False)
            one_file_df[FILE_NAME] = _file_name  # 增加文件名关联，以用于分割预测结果。
            extra_data_df[FILE_NAME] = _file_name  # 增加文件名关联，以用于分割预测结果。
            test_data_list.append(one_file_df)
            extra_data_list.append(extra_data_df)
        test_df = pd.concat(test_data_list, axis=0)
        extra_df = pd.concat(extra_data_list, axis=0)
        return test_df, extra_df

    @staticmethod
    def __split_and_generate_predict_result(result_df: pd.DataFrame, extra_df, out_result_folder: str):
        print('# Generating Predict Result To:', out_result_folder)
        # result结果至少包括三列，一列ts，一列predict，一列文件名
        unique_file_names = result_df[FILE_NAME].unique().tolist()
        for _file_name in tqdm(unique_file_names):
            file_result_df = result_df[result_df[FILE_NAME] == _file_name]
            extra_result_df = extra_df[extra_df[FILE_NAME] == _file_name]
            final_df = pd.concat([file_result_df, extra_result_df], axis=0)
            final_df.sort_values(by=TimeStamp)
            final_df.reset_index(drop=True, inplace=True)
            out_result_file = os.path.join(out_result_folder, _file_name)
            file_result_df.to_csv(out_result_file, index=False)

    def train_predict_in_cross_folders(self, out_predict_result_folder, with_train=True, test_data_folder=None):
        for _folder, _file_names in self.folder_map_file_names.items():
            print()
            print(f'# Running Folder: {_folder}/{len(self.folder_map_file_names)}')
            print()
            if with_train:
                train_folder_list = [x for x in self.folder_map_file_names.keys() if x != _folder]
                train_df = self.__get_train_data_by_folder(train_folder_list)
                if _folder == -1:  # TODO: 仅仅此次
                    self.__train_model_in_one_folder(train_df, retrain=False)
                else:
                    self.__train_model_in_one_folder(train_df, retrain=True)
            test_folder_list = [_folder]
            test_df, extra_df = self.__get_test_data_by_folder(test_folder_list)
            print('$$$$$$:', test_df.columns.tolist())
            result_df = self.__predict_model_in_one_folder(test_df, extra_df)
            self.__split_and_generate_predict_result(result_df, extra_df, out_predict_result_folder)
            print('run test predict ing...')
            test_df, extra_df = self.__get_test_data_by_folder_path(test_data_folder)
            result_df = self.__predict_model_in_one_folder(test_df, extra_df)
            self.__split_and_generate_predict_result(result_df, extra_df, os.path.join(out_predict_result_folder, 'test', f'folder_{_folder}'))


if __name__ == '__main__':
    model_runner = ModelRunner(model_type=RuleModel())
    model_runner.train_predict_in_cross_folders(with_train=False)

