from tqdm import tqdm

from Config import *
from ModelHub.BaseModel import BaseModel


class RuleModel(BaseModel):
    MODEL_NAME = MnRuleModel

    def __init__(self):
        super().__init__()
        self.rule_set = set()

    def fit(self, train_data, retain=False):
        pass

    def set_rules(self, rule_list):
        self.rule_set = set(rule_list)
        assert len(self.rule_set) == len(rule_list)

    def predict(self, test_data, ret_each_rule=RETURN_EXTRA):
        print(f'# Running Rule Predict...')
        # 预测特征表格每一行结果即可, 输出结果至少包括三列，一列ts，一列predict，一列文件名
        if len(self.rule_set) == 0:
            raise RuntimeError('Please Set Rule Before Predict!!!')
        print('Rule Count:', len(self.rule_set))
        print('Rule Setting:', self.rule_set)
        for _rule in tqdm(self.rule_set):
            if _rule == RuTempMin:
                test_data[_rule] = test_data[FeTempMin].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuCurrMin:
                test_data[_rule] = test_data[FeCurrentMin].apply(lambda x: 1 if x < 5000 else 0)
            elif _rule == RuTempDiff:
                test_data[_rule] = test_data[FeTempDiff].apply(lambda x: 1 if x > 100 else 0)
            elif _rule == RuCurrDiff:
                test_data[_rule] = test_data[FeCurrentDiff].apply(lambda x: 1 if x > 6000 else 0)
            elif _rule == RuTempStd:
                test_data[_rule] = test_data[FeTempStd].apply(lambda x: 1 if x > 10 else 0)
            elif _rule == RuCurrStd:
                test_data[_rule] = test_data[FeCurrentStd].apply(lambda x: 1 if x > 1500 else 0)
            elif _rule == RuTempSkew:
                test_data[_rule] = test_data[FeTempSkew].apply(lambda x: 1 if x < -20 else 0)
            elif _rule == RuTempKurt:
                test_data[_rule] = test_data[FeTempKurt].apply(lambda x: 1 if x > 500 else 0)
            elif _rule == RuTxP0Min:
                test_data[_rule] = test_data[FeTxP0Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuRxP0Min:
                # print('FeRxP0Min:', test_data[FeTxP0Min].sum(), len(test_data[FeTxP0Min]))
                test_data[_rule] = test_data[FeRxP0Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuTxP0Diff:
                test_data[_rule] = test_data[FeTxP0Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuRxP0Diff:
                test_data[_rule] = test_data[FeRxP0Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuTxP0Std:
                test_data[_rule] = test_data[FeTxP0Std].apply(lambda x: 1 if x > 500 else 0)
            elif _rule == RuRxP0Std:
                test_data[_rule] = test_data[FeRxP0Std].apply(lambda x: 1 if x > 500 else 0)
            elif _rule == RuTxRxP0Skew:
                test_data[_rule] = list(map(lambda x, y: 1 if x < -10 and y < -10 else 0, test_data[FeTxP0Skew], test_data[FeRxP0Skew]))
            elif _rule == RuTxRxP0Kurt:
                test_data[_rule] = list(map(lambda x, y: 1 if x > 500 and y > 500 else 0, test_data[FeTxP0Kurt], test_data[FeRxP0Kurt]))
            elif _rule == RuRxP1Min:
                test_data[_rule] = test_data[FeRxP1Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuRxP2Min:
                test_data[_rule] = test_data[FeRxP2Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuRxP1Diff:
                test_data[_rule] = test_data[FeRxP1Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuRxP2Diff:
                test_data[_rule] = test_data[FeRxP2Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuRxP1Std:
                test_data[_rule] = test_data[FeRxP1Std].apply(lambda x: 1 if x > 200 else 0)
            elif _rule == RuRxP2Std:
                test_data[_rule] = test_data[FeRxP2Std].apply(lambda x: 1 if x > 200 else 0)
            elif _rule == RuRxP1P2Skew:
                test_data[_rule] = list(map(lambda x, y: 1 if x < -20 and y < -20 else 0, test_data[FeRxP1Skew], test_data[FeRxP1Skew]))
            elif _rule == RuRxP3Min:
                test_data[_rule] = test_data[FeRxP3Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuRxP4Min:
                test_data[_rule] = test_data[FeRxP4Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuRxP3Diff:
                test_data[_rule] = test_data[FeRxP3Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuRxP4Diff:
                test_data[_rule] = test_data[FeRxP4Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuRxP3Std:
                test_data[_rule] = test_data[FeRxP3Std].apply(lambda x: 1 if x > 200 else 0)
            elif _rule == RuRxP4Std:
                test_data[_rule] = test_data[FeRxP4Std].apply(lambda x: 1 if x > 200 else 0)
            elif _rule == RuRxP3P4Skew:
                test_data[_rule] = list(map(lambda x, y: 1 if x < -20 and y < -20 else 0, test_data[FeRxP3Skew], test_data[FeRxP3Skew]))
            elif _rule == RuTxP1Min:
                test_data[_rule] = test_data[FeTxP1Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuTxP2Min:
                test_data[_rule] = test_data[FeTxP2Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuTxP1Diff:
                test_data[_rule] = test_data[FeTxP1Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuTxP2Diff:
                test_data[_rule] = test_data[FeTxP2Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuTxP1Std:
                test_data[_rule] = test_data[FeTxP1Std].apply(lambda x: 1 if x > 100 else 0)
            elif _rule == RuTxP2Std:
                test_data[_rule] = test_data[FeTxP2Std].apply(lambda x: 1 if x > 100 else 0)
            elif _rule == RuTxP3Min:
                test_data[_rule] = test_data[FeTxP3Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuTxP4Min:
                test_data[_rule] = test_data[FeTxP4Min].apply(lambda x: 1 if x < 0 else 0)
            elif _rule == RuTxP3Diff:
                test_data[_rule] = test_data[FeTxP3Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuTxP4Diff:
                test_data[_rule] = test_data[FeTxP4Diff].apply(lambda x: 1 if x > 1000 else 0)
            elif _rule == RuTxP3Std:
                test_data[_rule] = test_data[FeTxP3Std].apply(lambda x: 1 if x > 100 else 0)
            elif _rule == RuTxP4Std:
                test_data[_rule] = test_data[FeTxP4Std].apply(lambda x: 1 if x > 100 else 0)
            else:
                raise RuntimeError('Some Error Happen!')
        test_data_cols = set(test_data.columns.tolist())
        # print('Match:', self.rule_set - test_data_cols)
        print('# Rule Result Summing...')
        test_data[RuleMatchCnt] = 0
        for _rule in tqdm(RULE_SET):
            test_data[RuleMatchCnt] += test_data[_rule]
        test_data[Predict] = test_data[RuleMatchCnt].apply(lambda x: 1 if x > 0 else 0)
        ret_cols = [TimeStamp, Predict, FILE_NAME]
        test_data[TimeStamp] = test_data[NTimeStamp]
        if ret_each_rule:
            for _rule in RULE_SET:
                ret_cols.append(_rule)
        return test_data[ret_cols]

