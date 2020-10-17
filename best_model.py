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
    dataset_path = 'data_digits_8_vs_9_noisy'
    x_train_df = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_train_NF = x_train_df.values
    N, F = x_train_NF.shape

    y_train_d = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_train_N = y_train_d.values.reshape((N,))

    x_valid_dm = pd.read_csv(os.path.join(dataset_path, 'x_valid.csv'))
    x_valid_MF = x_valid_dm.values
    M = x_valid_MF.shape[0]

    y_valid_d = pd.read_csv(os.path.join(dataset_path, 'y_valid.csv'))
    y_valid_M = y_valid_d.values.reshape((M,))

    # training_BCEs = []
    # valid_BCEs = []
    training_ERs = []
    valid_ERs = []

    C_grid = np.logspace(-9, 6, 31)

    classifier = sklearn.linear_model.LogisticRegression(C=0.01, solver='lbfgs', max_iter=1000)
    classifier.fit(x_train_NF, y_train_N)

    coefficients = classifier.coef_
    image = coefficients.reshape((28,28))
    plt.imshow([[0, 1], [-0.5, 0.5]], cmap='RdYlBu', vmin=-0.5, vmax=0.5)
    plt.show()


    # yhat_M_valid = classifier.predict(x_valid_MF) >= 0.5
    #
    # print(y_valid_M)
    # print(yhat_M_valid)
    #
    # FNs = []
    # FPs = []
    #
    # y_valid_M_bools = y_valid_M >= 0.5
    # for i in range(len(y_valid_M)):
    #     if y_valid_M_bools[i] == True and yhat_M_valid[i] == False:
    #         FNs.append(i)
    #     elif y_valid_M_bools[i] == False and yhat_M_valid[i] == True:
    #         FPs.append(i)
    # print("FNs:", FNs)
    # print("FPs:", FPs)



if __name__ == '__main__':
    train_i_range()