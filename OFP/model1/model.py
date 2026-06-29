import os
import pandas as pd
import xgboost as xgb

def prediction_result(file_folder, model1_file):
    # Load the XGBoost model
    loaded_xgb_model = xgb.Booster()
    loaded_xgb_model.load_model(model1_file)
    model1_threshold = 0.5
    model2_threshold = 0.1

    # Ensure the output directory exists
    output_folder = './output_test'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the directory
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            
            # Read the CSV file into a DataFrame
            data = pd.read_csv(file_path)

            # Separate the timestamp
            timestamps = data['timestamp']
            data = data.drop(columns=['timestamp'])

            # Prepare the data for prediction
            dtest = xgb.DMatrix(data)

            # Make predictions
            predictions = loaded_xgb_model.predict(dtest)
            predictions_binary = [1 if y >= model1_threshold else 0 for y in predictions]

            # Combine predictions with timestamps
            result = pd.DataFrame({'timestamp': timestamps, 'predict': predictions_binary})

            # Save the result to a new CSV file
            output_path = os.path.join(output_folder, filename)
            result.to_csv(output_path, index=False)

if __name__ == '__main__':
    # 默认使用model1
    file_folder = './test'
    model1_file = 'xgboost_model_optical_original_features_ahead_120.json'
    model2_file = 'xgboost_model_optical_original_features_ahead_120_all_train_set_drop_anomaly_drop_NaN.json'
    prediction_result(file_folder, model1_file)


