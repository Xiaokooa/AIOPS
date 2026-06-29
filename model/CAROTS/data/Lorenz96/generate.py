import os

import numpy as np

from simu_data import simulate_lorenz_96
from multivariate_generator import MultivariateDataGenerator
import pickle


def save_outliers(data, method, var_num=10, ratio=0.01, factor=None, radius=None, save_path='', dataset_type='test'):
    data = data.transpose()
    anomaly_generator = MultivariateDataGenerator(data)

    func = getattr(anomaly_generator, method)
    if factor is None:
        outlier_data, outlier_labels, outlier_info = func(var_num=var_num, ratio=ratio, radius=radius)
    else:
        outlier_data, outlier_labels, outlier_info = func(var_num=var_num, ratio=ratio, factor=factor, radius=radius)
        
    outlier_data = outlier_data.transpose()
    
    data_file = os.path.join(save_path, f'{dataset_type}_{method}_factor{factor}.npy')
    labels_file = os.path.join(save_path, f'{dataset_type}_{method}_factor{factor}_labels.npy')
    info_file = os.path.join(save_path, f'{dataset_type}_{method}_factor{factor}_info.pkl')
    
    np.save(data_file, outlier_data)
    np.save(labels_file, outlier_labels)
    with open(info_file, 'wb') as f:
        pickle.dump(outlier_info, f)
    
    print(f"Outlier data saved to: {data_file}")
    print(f"Outlier labels saved to: {labels_file}")
    print(f"Outlier info saved to: {info_file}")
    
    return outlier_data, outlier_labels


if __name__ == '__main__':
    length = 40000
    save_path = './'

    np.random.seed(0)  # Fix the seed for reproducibility

    data, GC = simulate_lorenz_96(p=128, T=length, seed=0)

    train = data[:(length // 2)]
    test = data[(length // 2):]

    np.save(os.path.join(save_path, 'train.npy'), train)
    np.save(os.path.join(save_path, 'GC.npy'), GC)

    for factor in (1.0, 2.0, 3.0, 4.0):
        save_outliers(test, 'point_global_outliers', factor=factor, save_path=save_path, dataset_type='test')
        save_outliers(test, 'point_contextual_outliers', factor=factor, radius=5, save_path=save_path, dataset_type='test')
        save_outliers(test, 'collective_trend_outliers', factor=factor, radius=5, save_path=save_path, dataset_type='test')

    save_outliers(test, 'collective_global_outliers', radius=5, save_path=save_path, dataset_type='test')
