import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def train_i_range():
    dataset_path = 'data_sneaker_vs_sandal'
    x_all_d = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_all = x_all_d.values
    A,F = x_all.shape

    x_train_NF = x_all[:9000]
    N = 9000
    x_valid_MF = x_all[9000:]
    M = 3000

    y_all_d = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_all = y_all_d.values.reshape((A,))
    y_train_N = y_all[:9000]
    y_valid_M = y_all[9000:]

    baseline = sklearn.linear_model.LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    baseline.fit(x_train_NF, y_train_N)
    baseline_err = sklearn.metrics.zero_one_loss(y_valid_M, baseline.predict(x_valid_MF) >= 0.5)
    print(baseline_err)


def test_C_vals(x_train_NF, y_train_N, x_valid_MF, y_valid_M):
    C_grid = np.logspace(-9, 6, 31)

    training_ERs = []
    valid_ERs = []

    for C in C_grid:
        classifier = sklearn.linear_model.LogisticRegression(C=C, solver='lbfgs', max_iter=1000)
        classifier.fit(x_train_NF, y_train_N)
        yproba1_N_train = classifier.predict(x_train_NF)
        # training_BCEs.append(sklearn.metrics.log_loss(y_train_N, yproba1_N_train))
        training_ERs.append(sklearn.metrics.zero_one_loss(y_train_N, yproba1_N_train >= 0.5))

        yproba1_M_valid = classifier.predict(x_valid_MF)
        # valid_BCEs.append(sklearn.metrics.log_loss(y_valid_M, yproba1_M_valid))
        valid_ERs.append(sklearn.metrics.zero_one_loss(y_valid_M, yproba1_M_valid >= 0.5))

    plt.plot(np.log10(C_grid), training_ERs, 'b:', label='train err')
    plt.plot(np.log10(C_grid), valid_ERs, 'r:', label='valid err')
    plt.ylabel('Error rate')
    plt.xlabel('log_{10} C')
    plt.title('Error Rate vs Penalty strength')
    plt.legend(loc="upper right")

    plt.show()

    print("Best C value is:", C_grid[np.argmin(valid_ERs)])



if __name__ == '__main__':
    train_i_range()