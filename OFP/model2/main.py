import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

from Config import *
from FeatureExtractor import FeatureExtractor
from FileData import FileData
from ModelHub.RuleModel import RuleModel
from ModelHub.TreeModel import TreeModel
from ModelHub.MutantModel import MutantModel
from Tools.EvaluateResult import evaluate_result
from Tools.TrainAndPredictResult import ModelRunner


def extract_all_features(csv_folder, out_feature_folder, default_features=False):
    file_list = os.listdir(csv_folder)
    i = 0
    for _file_name in tqdm(file_list):
        i += 1
        # if i < 1920 and 'pos' in out_feature_folder: # TODO
        #     continue

        file_data = FileData(csv_folder, _file_name)
        df = file_data.get_data(with_column_map=True)
        sn = file_data.get_sn()
        df = df[~df[NTemperature].isna()]
        df.reset_index(drop=True, inplace=True)
        feature_extractor = FeatureExtractor(df)
        if default_features:
            feature_df = feature_extractor.get_feature_by_default(sn, win_size=5)
        else:
            feature_names = feature_extractor.generate_feature_names()
            feature_df = feature_extractor.get_feature_by_series(feature_names, sn)
        out_feature_file_path = os.path.join(out_feature_folder, f'{FEATURE_FILE_PREFIX}{_file_name}')
        feature_df.to_csv(out_feature_file_path, index=False, mode='w')
        # break


def train_and_predict(model_name):
    if model_name in {MnMutantModel}:
        # MUTANT模型
        model = MutantModel()
        model.set_features(feature_list=ORIGIN_FEATURE_LIST)  # TODO
        model_runner = ModelRunner(model_type=model)
        model_runner.train_predict_in_cross_folders(MUTANT_PREDICT_FOLDER, with_train=True)
    elif model_name == MnRuleModel:
        # 专家规则模型
        model = RuleModel()
        model.set_rules(rule_list=RULE_SET)
        model_runner = ModelRunner(model_type=model)
        model_runner.train_predict_in_cross_folders(RULE_PREDICT_FOLDER, with_train=False, test_data_folder=FEATURE_TEST_FOLDER)
    else:
        # 树模型
        model = TreeModel()
        # model.set_features(feature_list=FEATURE_LIST)  # TODO
        model.set_features(feature_list=DEFAULT_FEATURE_LIST)  # TODO
        model_runner = ModelRunner(model_type=model)
        model_runner.train_predict_in_cross_folders(TREE_PREDICT_FOLDER, with_train=True, test_data_folder=FEATURE_TEST_FOLDER)


def evaluate_and_report(model_name):
    if model_name in {MnMutantModel}:
        evaluate_result(MUTANT_PREDICT_FOLDER, LABEL_FOLDER, MUTANT_EVALUATE_FOLDER)
    elif model_name == MnRuleModel:
        evaluate_result(RULE_PREDICT_FOLDER, LABEL_FOLDER, RULE_EVALUATE_FOLDER)
    else:
        evaluate_result(TREE_PREDICT_FOLDER, LABEL_FOLDER, TREE_EVALUATE_FOLDER)


if __name__ == '__main__':
    _model_name = MnTreeModel  # TODO: 选择并设置模型  MnRuleModel, MnTreeModel, MnMutantModel
    if _model_name == MnTreeModel:
        default_features = True
    else:
        default_features = False
    root_folder = r'd:\new_data'
    # root_folder = r'C:\Users\z00381790\Desktop\光模块故障预测比赛\new_data\debug_data'
    _pos_folder = os.path.join(root_folder, 'pos_samples')
    _neg_folder = os.path.join(root_folder, 'neg_samples')
    _test_folder = os.path.join(root_folder, 'test')
    _out_feature_path = os.path.join(root_folder, 'feature_data')
    # _out_feature_path = os.path.join(root_folder, 'feature_data_default')  # TODO
    # 将正负样本特征分别提取到对应文件夹
    extract_all_features(_pos_folder, os.path.join(_out_feature_path, 'pos'), default_features=default_features)
    extract_all_features(_neg_folder, os.path.join(_out_feature_path, 'neg'), default_features=default_features)
    extract_all_features(_test_folder, os.path.join(_out_feature_path, 'test'), default_features=default_features)
    # 训练与预测模型
    train_and_predict(_model_name)
    # 评估结果
    evaluate_and_report(_model_name)

