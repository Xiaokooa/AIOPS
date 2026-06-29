import os
from Golbal import *


MODEL_NAME = MnTreeModel  # TODO:TODO: 换模型需要修改
ROOT_FOLDER = r'D:\new_data'
# ROOT_FOLDER = r'C:\Users\光模块故障预测比赛\new_data\debug_data'
# 原始csv相关
POS_FOLDER = os.path.join(ROOT_FOLDER, 'pos_samples')
NEG_FOLDER = os.path.join(ROOT_FOLDER, 'neg_samples')
POS_CSV_LIST = os.listdir(POS_FOLDER)
NEG_CSV_LIST = os.listdir(NEG_FOLDER)
ALL_CSV_LIST = [*POS_CSV_LIST, *NEG_CSV_LIST]
# 交叉验证相关
TRAIN_TEST_SET_FOLDER = os.path.join(ROOT_FOLDER, 'train_test_set')
TRAIN_TEST_SET_INDEX_CSV = os.path.join(TRAIN_TEST_SET_FOLDER, 'train_test_set_index.csv')
# 特征相关
if MODEL_NAME == MnTreeModel:
    FEATURE_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'feature_data_default')  # TODO: 换模型需要修改
else:  # Rule
    FEATURE_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'feature_data')
FEATURE_POS_FOLDER = os.path.join(FEATURE_ROOT_FOLDER, 'pos')
FEATURE_NEG_FOLDER = os.path.join(FEATURE_ROOT_FOLDER, 'neg')
FEATURE_TEST_FOLDER = os.path.join(FEATURE_ROOT_FOLDER, 'test')
# 预测与评估相关
RESULT_ROOT_FOLDER = os.path.join(ROOT_FOLDER, 'predict_result')
RULE_PREDICT_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'rule_predict')
RULE_EVALUATE_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'rule_evaluate')
RULE_FILE2RULE_FILE = os.path.join(RESULT_ROOT_FOLDER, 'rule_file2rule.csv')
RULE_EACH_RULE_RESULT_FILE = os.path.join(RESULT_ROOT_FOLDER, 'rule_each_rule_result.csv')
# TREE_PREDICT_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'folder_cross_split_qiaoyu2')  #TODO
# TREE_PREDICT_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'tree_predict_merge_qiaoyu')
TREE_PREDICT_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'tree_predict_default')
TREE_EVALUATE_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'tree_evaluate_default')
MUTANT_PREDICT_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'mutant_predict')
MUTANT_EVALUATE_FOLDER = os.path.join(RESULT_ROOT_FOLDER, 'mutant_evaluate')

# 专家规则
RULE_SET = {
    RuTempMin,
    RuCurrMin,
    RuTempDiff,
    RuCurrDiff,
    RuTempStd,
    RuCurrStd,
    # RuTempSkew,
    RuTempKurt,
    RuTxP0Min,
    RuRxP0Min,
    RuTxP0Diff,
    RuRxP0Diff,
    RuTxP0Std,
    RuRxP0Std,
    # RuTxRxP0Skew,
    RuTxRxP0Kurt,
    RuRxP1Min,
    RuRxP2Min,
    RuRxP1Diff,
    RuRxP2Diff,
    RuRxP1Std,
    RuRxP2Std,
    # RuRxP1P2Skew,
    RuRxP3Min,
    RuRxP4Min,
    RuRxP3Diff,
    RuRxP4Diff,
    RuRxP3Std,
    RuRxP4Std,
    # RuRxP3P4Skew,
    RuTxP1Min,
    RuTxP2Min,
    RuTxP1Diff,
    RuTxP2Diff,
    RuTxP1Std,
    RuTxP2Std,
    RuTxP3Min,
    RuTxP4Min,
    RuTxP3Diff,
    RuTxP4Diff,
    RuTxP3Std,
    RuTxP4Std,
}
# 标签
LABEL_FOLDER = os.path.join(ROOT_FOLDER, 'training')
# 训练特征设置
ORIGIN_FEATURE_MODELS = {MnMutantModel}
ORIGIN_FEATURE_LIST = [x for x in ColumnNameMap.values() if x not in {NAnomaly}]  # TODO
DEFAULT_FEATURE_LIST = [x for x in ColumnNameMap.values() if x not in {NAnomaly}]  # TODO
DEFAULT_FEATURE_LIST.extend([f"FeCo{NCurrent}{NTemperature}", f"FeCo{NCurrent}{NTxPower}", f"FeCo{NCurrent}{NRxPower}", f"FeCo{NTxPower}{NRxPower}"])
DEFAULT_FEATURE_LIST.extend([f"Fe{NTxPower}-Max", f"Fe{NTxPower}-Min", f"Fe{NRxPower}-Max", f"Fe{NRxPower}-Min"])
DEFAULT_FEATURE_LIST.extend([NTimeStampDelta])
FEATURE_LIST = [
     'FeTsMin',
     'FeTsDiff',
     'FeTsMax',
     'FeTsSkew',
     'FeTsKurt',
     'FeTsStd',
     'FeTempMin',
     'FeTempDiff',
     'FeTempMax',
     'FeTempSkew',
     'FeTempKurt',
     'FeTempStd',
     'FeCurrMin',
     'FeCurrDiff',
     'FeCurrMax',
     'FeCurrSkew',
     'FeCurrKurt',
     'FeCurrStd',
     'FeTxP0Min',
     'FeTxP0Diff',
     'FeTxP0Max',
     'FeTxP0Skew',
     'FeTxP0Kurt',
     'FeTxP0Std',
     'FeRxP0Min',
     'FeRxP0Diff',
     'FeRxP0Max',
     'FeRxP0Skew',
     'FeRxP0Kurt',
     'FeRxP0Std',
     'FeRxP1Min',
     'FeRxP1Diff',
     'FeRxP1Max',
     'FeRxP1Skew',
     'FeRxP1Kurt',
     'FeRxP1Std',
     'FeRxP2Min',
     'FeRxP2Diff',
     'FeRxP2Max',
     'FeRxP2Skew',
     'FeRxP2Kurt',
     'FeRxP2Std',
     'FeRxP3Min',
     'FeRxP3Diff',
     'FeRxP3Max',
     'FeRxP3Skew',
     'FeRxP3Kurt',
     'FeRxP3Std',
     'FeRxP4Min',
     'FeRxP4Diff',
     'FeRxP4Max',
     'FeRxP4Skew',
     'FeRxP4Kurt',
     'FeRxP4Std',
     'FeTxP1Min',
     'FeTxP1Diff',
     'FeTxP1Max',
     'FeTxP1Skew',
     'FeTxP1Kurt',
     'FeTxP1Std',
     'FeTxP2Min',
     'FeTxP2Diff',
     'FeTxP2Max',
     'FeTxP2Skew',
     'FeTxP2Kurt',
     'FeTxP2Std',
     'FeTxP3Min',
     'FeTxP3Diff',
     'FeTxP3Max',
     'FeTxP3Skew',
     'FeTxP3Kurt',
     'FeTxP3Std',
     'FeTxP4Min',
     'FeTxP4Diff',
     'FeTxP4Max',
     'FeTxP4Skew',
     'FeTxP4Kurt',
     'FeTxP4Std']
# 预测设置
RETURN_EXTRA = True  # 是否返回附加信息

