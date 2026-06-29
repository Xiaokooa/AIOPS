import pandas as pd

from Golbal import *
from Config import *
from pprint import pprint
from scipy.stats import pearsonr
from collections import defaultdict
from numpy import corrcoef


class FeatureExtractor:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def __get_in_feature_name(out_feature: str):
        for c in ColumnNameMap.values():
            if c in out_feature:
                return c

    @staticmethod
    def __get_in_feature_type(out_feature: str):
        for m in MethodHub:
            if out_feature.endswith(m):
                return m

    @staticmethod
    def generate_feature_names(by_default=False):
        if by_default:
            return DEFAULT_FEATURE_LIST
        feature_name_list = []
        for c in ColumnNameMap.values():
            if c not in {NAnomaly}:
                for m in MethodHub:
                    feature_name = f"Fe{c}{m}"
                    feature_name_list.append(feature_name)
        return feature_name_list

    def get_feature_by_series(self, feature_names, sn):
        # print(self.df.columns)
        data_dict = dict()
        # step 是否是第一个step也有区别
        for fn in feature_names:
            cn = self.__get_in_feature_name(fn)
            mt = self.__get_in_feature_type(fn)
            if mt == MeDiff:
                continue
            f_list = []
            last_value = None
            for i in self.df.index:
                cur_df_v = self.df[cn][i]
                if i == 0:
                    if mt in {MeMin, MeMax}:
                        f_value = cur_df_v
                    elif mt in {MeStd, MeDiff}:
                        f_value = 0
                    elif mt == MeSkew:
                        f_value = self.df[cn][:i+1].skew()
                    elif mt == MeKurt:
                        f_value = self.df[cn][:i + 1].kurt()
                    else:
                        raise KeyError(f'{mt} not found!')
                else:
                    if mt == MeMin:
                        f_value = min(cur_df_v, last_value)
                    elif mt == MeMax:
                        f_value = max(cur_df_v, last_value)
                    elif mt == MeStd:
                        f_value = self.df[cn][:i+1].std()
                    elif mt == MeSkew:
                        f_value = self.df[cn][:i+1].skew()
                    elif mt == MeKurt:
                        f_value = self.df[cn][:i + 1].kurt()
                    else:
                        raise KeyError(f'{mt} not found!')
                f_list.append(f_value)
                last_df_v = cur_df_v
                last_value = f_value
            data_dict[fn] = f_list
        feat_df = pd.DataFrame(data_dict)
        # print(feat_df.columns.tolist())
        # print(feature_names)
        # 补充计算diff
        for fn in feature_names:
            cn = self.__get_in_feature_name(fn)
            mt = self.__get_in_feature_type(fn)
            if mt == MeDiff:
                # print(cn, mt)
                cn_min = f"Fe{cn}{MeMin}"
                cn_max = f"Fe{cn}{MeMax}"
                feat_df[fn] = feat_df[cn_max] - feat_df[cn_min]
        # feat_df[NAnomaly] = self.df[NAnomaly]
        feat_df['sn'] = sn
        feat_df['step'] = [x for x in range(1, len(self.df)+1)]
        if NAnomaly in self.df.columns.tolist():
            feat_df[Label] = 1 if self.df[NAnomaly].sum() > 0 else 0
        # 把原始数据也放进去，方便对比
        for c in self.df.columns:
            feat_df[c] = self.df[c]
        feat_df.fillna(NaDefault, inplace=True)  # 计算偏度、峰度时可能会有null
        return feat_df

    def get_feature_by_default(self, sn, win_size=5):
        df = self.df
        # if len(self.df) < win_size:
        #     return pd.DataFrame()
        data_dict = defaultdict(list)
        # step 是否是第一个step也有区别
        for i in self.df.index:
            if i < win_size - 1:
                data_dict[f"FeCo{NCurrent}{NTemperature}"].append(99)
                data_dict[f"FeCo{NCurrent}{NTxPower}"].append(-99)
                data_dict[f"FeCo{NCurrent}{NRxPower}"].append(-99)
                data_dict[f"FeCo{NTxPower}{NRxPower}"].append(99)
            else:
                win_start = i - win_size + 1
                win_end = i + 1
                # print(self.df[NCurrent][win_start:win_end], sn)
                # print(self.df[NTemperature][win_start:win_end], sn)
                data_dict[f"FeCo{NCurrent}{NTemperature}"].append(pearsonr(self.df[NCurrent][win_start:win_end], self.df[NTemperature][win_start:win_end])[0])
                data_dict[f"FeCo{NCurrent}{NTxPower}"].append(pearsonr(self.df[NCurrent][win_start:win_end], self.df[NTxPower][win_start:win_end])[0])
                data_dict[f"FeCo{NCurrent}{NRxPower}"].append(pearsonr(self.df[NCurrent][win_start:win_end], self.df[NRxPower][win_start:win_end])[0])
                data_dict[f"FeCo{NTxPower}{NRxPower}"].append(pearsonr(self.df[NTxPower][win_start:win_end], self.df[NRxPower][win_start:win_end])[0])
            # for col in ColumnNameMap.values():
            #     if col not in {NTimeStamp, NAnomaly}:
            #         data_dict[f"{col}Max"].append(self.df[col][win_start:win_end].max())
            #         data_dict[f"{col}Min"].append(self.df[col][win_start:win_end].min())
            data_dict[f"Fe{NTxPower}-Max"].append(df[NTxPower][i] - max(df[NTxPower1][i], df[NTxPower2][i], df[NTxPower3][i], df[NTxPower4][i]))
            data_dict[f"Fe{NTxPower}-Min"].append(df[NTxPower][i] - min(df[NTxPower1][i], df[NTxPower2][i], df[NTxPower3][i], df[NTxPower4][i]))
            data_dict[f"Fe{NRxPower}-Max"].append(df[NRxPower][i] - max(df[NRxPower1][i], df[NRxPower2][i], df[NRxPower3][i], df[NRxPower4][i]))
            data_dict[f"Fe{NRxPower}-Min"].append(df[NRxPower][i] - min(df[NRxPower1][i], df[NRxPower2][i], df[NRxPower3][i], df[NRxPower4][i]))
        feat_df = pd.DataFrame(data_dict)
        # print(feat_df.columns.tolist())
        # print(feature_names)
        # 补充计算diff
        # for fn in feature_names:
        #     cn = self.__get_in_feature_name(fn)
        #     mt = self.__get_in_feature_type(fn)
        #     if mt == MeDiff:
        #         # print(cn, mt)
        #         cn_min = f"Fe{cn}{MeMin}"
        #         cn_max = f"Fe{cn}{MeMax}"
        #         feat_df[fn] = feat_df[cn_max] - feat_df[cn_min]
        # feat_df[NAnomaly] = self.df[NAnomaly]
        feat_df['sn'] = sn
        feat_df['step'] = [x for x in range(1, len(self.df)+1)]
        if NAnomaly in self.df.columns.tolist():
            feat_df[Label] = 1 if self.df[NAnomaly].sum() > 0 else 0
        # 把原始数据也放进去，方便对比
        for c in self.df.columns:
            feat_df[c] = self.df[c]
        feat_df.fillna(NaDefault, inplace=True)  # 计算偏度、峰度时可能会有null
        return feat_df


if __name__ == '__main__':
    # 打印所有特征名称
    feature_names = FeatureExtractor.generate_feature_names()
    pprint(feature_names)
