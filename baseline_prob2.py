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
from performance_metrics import (calc_mean_squared_error, calc_ACC, calc_TPR, 
                                 calc_PPV, calc_TNR, calc_NPV, calc_confusion_matrix_for_probas_and_threshold)

# train_static_val runs the standard model splitting data
# train_cv_model runs cross validation model

@ignore_warnings(category=ConvergenceWarning)
def train_static_val(estimator):
    x_train_NF, y_train_N, x_valid_MF, y_valid_M = get_std_data()

    estimator.fit(x_train_NF, y_train_N)
    
    err_train = sklearn.metrics.zero_one_loss(y_train_N, estimator.predict(x_train_NF) >= 0.5)
    err_valid = sklearn.metrics.zero_one_loss(y_valid_M, estimator.predict(x_valid_MF) >= 0.5)
    
    yproba1_train_N = estimator.predict_proba(x_train_NF)[:,1]
    yproba1_valid_M = estimator.predict_proba(x_valid_MF)[:,1]
    
    cm_df = calc_confusion_matrix_for_probas_and_threshold(y_train_N, yproba1_train_N, .5)
    print(cm_df)
    print("TPR: ", calc_TPR(y_train_N, yproba1_train_N >= 0.5))
    print("PPV: ", calc_PPV(y_train_N, yproba1_train_N >=0.5))
    print("TNR: ", calc_TNR(y_train_N, yproba1_train_N >=0.5))
    print("NPV: ", calc_NPV(y_train_N, yproba1_train_N >=0.5))
    print("Training Error:", err_train)
    print("Validation Error:", err_valid)

def train_cv_model(estimator):
    x_train_NF, y_train_N = get_cv_data()

    # estimator.fit(x_train_NF, y_train_N)
    #
    # coefficients = estimator.coef_
    # image = coefficients.reshape((28,28))
    # plt.imshow(image, cmap='RdYlBu', vmin=-0.5, vmax=0.5)
    # plt.show()
    
    train_err_K, valid_err_K = train_models_and_calc_scores_for_n_fold_cv(estimator, x_train_NF, y_train_N, 3, 1)
    err_train = np.mean(train_err_K)
    err_valid = np.mean(valid_err_K)
    
    print(err_train, err_valid)
    
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

def get_cv_data ():
    dataset_path = 'data_sneaker_vs_sandal'
    x_train_df = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_train_NF = x_train_df.values
    N, F = x_train_NF.shape

    y_train_d = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_train_N = y_train_d.values.reshape((N,))
    
    return x_train_NF, y_train_N

def get_std_data ():
    dataset_path = 'data_sneaker_vs_sandal'
    x_all_d = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
    x_all = x_all_d.values
    A, F = x_all.shape

    x_train_NF = x_all[:9000]
    N = 9000
    x_valid_MF = x_all[9000:]
    M = 3000

    y_all_d = pd.read_csv(os.path.join(dataset_path, 'y_train.csv'))
    y_all = y_all_d.values.reshape((A,))
    y_train_N = y_all[:9000]
    y_valid_M = y_all[9000:]
    
    return x_train_NF, y_train_N, x_valid_MF, y_valid_M

def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    
    train_error_per_fold = np.zeros(n_folds, dtype=np.float)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float)
    N, F = x_NF.shape
  
    tr_ids_per_f, te_ids_per_f = make_train_and_test_row_ids_for_n_fold_cv(
                                                        N,n_folds,random_state)

    for i, j, k in zip(tr_ids_per_f, te_ids_per_f, range(n_folds)):
        estimator.fit(np.take(x_NF, i, axis = 0), np.take(y_N, i))
        y_hat_tr = estimator.predict(np.take(x_NF, i, axis = 0))
        y_hat_te = estimator.predict(np.take(x_NF, j, axis = 0))
        
        np.put(train_error_per_fold, k, calc_mean_squared_error(np.take(y_N, i), y_hat_tr))
        np.put(test_error_per_fold, k, calc_mean_squared_error(np.take(y_N, j), y_hat_te))
         
    return train_error_per_fold, test_error_per_fold

def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
        
    if hasattr(random_state, 'rand'):
        random_state = random_state 
    else:
        random_state = np.random.RandomState(int(random_state))
    
    test_array = random_state.permutation(np.arange(n_examples))
    test_array = np.array_split(test_array, n_folds)

    train_array = list()
    for i in range(n_folds):
        train_array.append(np.setxor1d(np.arange(n_examples), test_array[i]))
    train_ids_per_fold = train_array
    test_ids_per_fold = test_array

    return train_ids_per_fold, test_ids_per_fold
    
def make_pipeline():
    function = lambda x : x**0.0001
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
         (
          'custom function', sklearn.preprocessing.FunctionTransformer(function)),
         ('logistic_regr', sklearn.linear_model.LogisticRegression(C=.01, solver='lbfgs', max_iter=1000)),
        ])
    return pipeline
        
if __name__ == '__main__':
    estimator = make_pipeline()
    # train_cv_model(estimator)
    train_static_val(estimator)
    