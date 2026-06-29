import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

def read_and_preprocess_data(directory):
    # Read all CSV files from the directory and concatenate them into a single DataFrame
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df['sn'] = os.path.splitext(filename)[0]
            data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    return data

def label_anomalies(data, time_windows):
    # Label data with anomalies ahead within specified time windows
    data_grouped = data.groupby('sn')
    for name, group in data_grouped:
        anomaly_timestamps = group[group['anomaly'] == 1]['timestamp']
        for anomaly_time in anomaly_timestamps:
            for window_name, window in time_windows.items():
                mask = (group['timestamp'] >= anomaly_time - window) & (group['timestamp'] < anomaly_time)
                group.loc[mask, f'anomaly_ahead_{window_name}'] = 1
        data_grouped.update(group)
    return data

def train_xgboost(X_train, y_train, X_test, y_test, num_rounds=100):
    # Set up and train the XGBoost model
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals_result = {}
    model = xgb.train(params, dtrain, num_rounds, evals=[(dtest, 'test')], evals_result=evals_result,
                      callbacks=[XGBoostProgressCallback(num_rounds)])
    return model, evals_result

def evaluate_model(model, dtest, y_test):
    # Predict and evaluate the model
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report

class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    # Callback to show progress bar during training
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="XGBoost Training Progress")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if epoch + 1 == self.pbar.total:
            self.pbar.close()
        return False

if __name__ == '__main__':
    file_folder = './train'
    data = read_and_preprocess_data(file_folder)
    time_windows = {
        '16_hours': 16 * 3600,
        '24_hours': 24 * 3600,
        '72_hours': 72 * 3600,
        '120_hours': 120 * 3600
    }
    data = label_anomalies(data, time_windows)
    X = data.drop(columns=['timestamp', 'anomaly', 'sn'] + [f'anomaly_ahead_{k}' for k in time_windows.keys()])
    y = data['anomaly_ahead_120_hours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model, _ = train_xgboost(X_train, y_train, X_test, y_test)
    accuracy, conf_matrix, class_report = evaluate_model(model, xgb.DMatrix(X_test), y_test)

    print("XGBoost Accuracy:", accuracy)
    print("XGBoost Confusion Matrix:\n", conf_matrix)
    print("XGBoost Classification Report:\n", class_report)
    model.save_model('xgboost_model.json')



