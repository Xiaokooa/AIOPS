import pandas as pd
from tqdm import tqdm

from Config import *
from LabelData import LabelData


class RuleModeResultFile:
    def __init__(self, folder_path, file_name):
        self.folder_path = folder_path
        self.file_name = file_name
        self.file_path = os.path.join(folder_path, file_name)
        self.sn = os.path.splitext(file_name)[0]
        # print(self.folder_path)
        # print(self.file_name, self.sn)
        self.df = self.read_data()
        # print(self.df.columns.tolist())
        # self.print_line_x(line_index=0)

    def read_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def get_sn(self):
        return self.sn

    def get_rule_map_result(self):
        rule_map_result = dict()
        for _rule in RULE_SET:
            hit_ts_cnt = self.df[_rule].sum()
            _result = hit_ts_cnt
            rule_map_result[_rule] = _result
        return rule_map_result

    def get_predict_hit_ts_cnt(self):
        return self.df[Predict].sum()


class RuleModeResultAnalyzer:
    def __init__(self, result_folder):
        self.result_folder = result_folder
        self.all_csv_list = self.__get_all_csv()

    def __get_all_csv(self):
        return os.listdir(self.result_folder)

    def get_each_file_each_rule(self, out_files_rules_file):
        data_list = []
        for _file in tqdm(self.all_csv_list):
            _path = os.path.join(self.result_folder, _file)
            if not os.path.isfile(_path):
                print(f'Warning: {_path} is not a file!!!')
                continue
            rule_file_obj = RuleModeResultFile(self.result_folder, _file)
            rule_map_result = rule_file_obj.get_rule_map_result()
            one_file_stat_dict = dict()
            one_file_stat_dict[FILE_NAME] = _file
            one_file_stat_dict.update(rule_map_result)
            one_file_stat_dict[Predict] = rule_file_obj.get_predict_hit_ts_cnt()
            data_list.append(one_file_stat_dict)
        pd.DataFrame(data=data_list).to_csv(out_files_rules_file, index=False)
        print(f"each_file_each_rule write to: {out_files_rules_file}")

    def stat_each_rule_result(self, files_rules_file, label_folder, out_each_rules_result_file):
        files_rules_file_df = pd.read_csv(files_rules_file)
        result_csv_list = os.listdir(label_folder)
        all_true_pos_files = set()
        for csv_name in tqdm(result_csv_list):
            one_label_obj = LabelData(label_folder, csv_name, label_column=Anomaly)
            true_label, true_ts = one_label_obj.get_label_and_first_ts()
            if true_label > 0:
                all_true_pos_files.add(csv_name)
        data_list = []
        for _rule in tqdm(RULE_SET):
            match_file_set = set(files_rules_file_df[files_rules_file_df[_rule]>0][FILE_NAME].tolist())
            one_rule_stat_dict = dict()
            one_rule_stat_dict['RuleName'] = _rule
            one_rule_stat_dict['HitFileCnt'] = len(all_true_pos_files & match_file_set)
            one_rule_stat_dict['MatchFileCnt'] = len(match_file_set)
            one_rule_stat_dict['PosFileCnt'] = len(all_true_pos_files)
            one_rule_stat_dict['Precision'] = one_rule_stat_dict['HitFileCnt']/one_rule_stat_dict['MatchFileCnt'] \
                if one_rule_stat_dict['MatchFileCnt'] > 0 else 0
            one_rule_stat_dict['Recall'] = one_rule_stat_dict['HitFileCnt'] / one_rule_stat_dict['PosFileCnt'] \
                if one_rule_stat_dict['PosFileCnt'] > 0 else 0
            one_rule_stat_dict['F1'] = 2*one_rule_stat_dict['Precision']*one_rule_stat_dict['Recall'] / \
                (one_rule_stat_dict['Precision'] + one_rule_stat_dict['Recall'])
            data_list.append(one_rule_stat_dict)
        pd.DataFrame(data=data_list).to_csv(out_each_rules_result_file, index=False)
        print(f"each_rule_result write to: {out_each_rules_result_file}")


if __name__ == '__main__':
    _analyzer = RuleModeResultAnalyzer(result_folder=RULE_PREDICT_FOLDER)
    _analyzer.get_each_file_each_rule(out_files_rules_file=RULE_FILE2RULE_FILE)
    _analyzer.stat_each_rule_result(files_rules_file=RULE_FILE2RULE_FILE, label_folder=LABEL_FOLDER,
                                    out_each_rules_result_file=RULE_EACH_RULE_RESULT_FILE)
