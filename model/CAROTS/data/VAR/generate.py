import os

import numpy as np

from simu_data import simulate_var
from multivariate_generator import MultivariateDataGenerator


if __name__ == '__main__':
    length = 40000
    save_path = './'

    data, beta, GC = simulate_var(p=128, T=length, lag=3, seed=0)

    train = data[:(length // 2)]
    test = data[(length // 2):]

    np.save(os.path.join(save_path, 'train.npy'), train)
    np.save(os.path.join(save_path, 'GC.npy'), GC)

    anomaly_generator = MultivariateDataGenerator(test.transpose())

    def save_outliers(method, factor=None, radius=None):
        func = getattr(anomaly_generator, method)
        if factor is None:
            outlier_data, outlier_labels = func(var_num=10, ratio=0.01, radius=radius)
        else:
            outlier_data, outlier_labels = func(var_num=10, ratio=0.01, factor=factor, radius=radius)
        outlier_data = outlier_data.transpose()
        np.save(os.path.join(save_path, f'test_{method}_factor{factor}.npy'), outlier_data)
        np.save(os.path.join(save_path, f'test_{method}_factor{factor}_labels.npy'), outlier_labels)

    for factor in (1.0, 2.0, 3.0, 4.0):
        save_outliers('point_global_outliers', factor=factor)
        save_outliers('point_contextual_outliers', factor=factor, radius=5)
        save_outliers('collective_trend_outliers', factor=factor, radius=5)

    save_outliers('collective_global_outliers', radius=5)
