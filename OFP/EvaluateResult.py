import math
import os

import pandas as pd
from tqdm import tqdm

from Config import *
from Golbal import *
from LabelData import LabelData


def generate_label_file():
    pass


def evaluate_result(result_folder, label_folder, out_evaluate_folder):
    print('# Evaluating result...')
    result_csv_list = os.listdir(result_folder)
    all_true_pos_sns = set()
    all_predict_pos_sns = set()
    lead_sec_list = []
    lead_pred_sn_list = list()
    all_cnt = 0
    for csv_name in tqdm(result_csv_list):
        _path = os.path.join(result_folder, csv_name)
        if not os.path.isfile(_path):
            print(f'Warning: {_path} is not a file!!!')
            continue
        one_result_obj = LabelData(result_folder, csv_name, label_column=Predict)
        one_label_obj = LabelData(label_folder, csv_name, label_column=Anomaly)
        predict_label, predict_ts = one_result_obj.get_label_and_first_ts()
        true_label, true_ts = one_label_obj.get_label_and_first_ts()
        sn = one_result_obj.get_sn()
        if predict_label > 0:
            if true_label == 0:
                all_predict_pos_sns.add(sn)
            elif true_ts > predict_ts:
                all_predict_pos_sns.add(sn)
        if true_label > 0:
            all_true_pos_sns.add(sn)
            if predict_label > 0:
                if true_ts > predict_ts:  # TODO：何为正确？
                # if true_ts <= predict_ts:
                    lead_sec = true_ts - predict_ts
                    lead_sec = abs(lead_sec)
                    lead_sec_list.append(lead_sec)
                    lead_pred_sn_list.append(sn)
        all_cnt += 1
    all_hit_sns = all_true_pos_sns & all_predict_pos_sns
    tp = len(all_hit_sns)
    fp = len(all_predict_pos_sns) - tp
    fn = len(all_true_pos_sns) - tp
    tn = all_cnt - tp - fp - fn
    accuracy = (tp + tn)/all_cnt
    precision = len(all_hit_sns) / len(all_predict_pos_sns) if len(all_predict_pos_sns) > 0 else 0
    recall = len(all_hit_sns) / len(all_true_pos_sns)
    f1_score = 2*precision*recall/(precision + recall)

    avg_lead_sec = sum(lead_sec_list) / len(all_true_pos_sns)
    avg_lead_hour = avg_lead_sec / SecInHour
    avg_lead_score = (math.exp(avg_lead_hour) - math.exp(-avg_lead_hour))/(math.exp(avg_lead_hour) + math.exp(-avg_lead_hour))

    min_lead_sec = min(lead_sec_list) if len(lead_sec_list) > 0 else 0
    min_lead_hour = min_lead_sec / SecInHour
    min_lead_score = (math.exp(min_lead_hour) - math.exp(-min_lead_hour))/(math.exp(min_lead_hour) + math.exp(-min_lead_hour))

    final_score = f1_score + avg_lead_score + min_lead_score + accuracy
    report_df = pd.DataFrame(data={
        'Item': ['final_score', 'f1_score', 'precision', 'recall', 'all_hit_cnt', 'all_predict_pos_cnt', 'all_true_pos_cnt', 'avg_lead_score', 'avg_lead_hour', 'min_lead_score', 'min_lead_hour', 'lead_pread_cnt', 'accuracy'],
        'Value': [final_score, f1_score, precision, recall, len(all_hit_sns), len(all_predict_pos_sns), len(all_true_pos_sns), avg_lead_score, avg_lead_hour, min_lead_score, min_lead_hour, len(lead_pred_sn_list), accuracy],
    })
    out_evaluate_file = os.path.join(out_evaluate_folder, 'evaluate_result.csv')
    report_df.to_csv(out_evaluate_file, index=False)
    print('Evaluation Report Write To:', out_evaluate_folder)
    # print(report_df)
    for i in report_df.index:
        print(f"{report_df['Item'][i]} = {report_df['Value'][i]}")
    return final_score


if __name__ == '__main__':
    evaluate_result(RULE_PREDICT_FOLDER, LABEL_FOLDER, RULE_EVALUATE_FOLDER)

