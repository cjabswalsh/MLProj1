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
def make_model():
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

    print("loaded data")
    feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
            ('orig', sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
            ])
    classifier = sklearn.linear_model.LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    pipeline = sklearn.pipeline.Pipeline([
        ('step1', feature_tfmr),
        ('step2', classifier)
        ])
    print("made pipeline")
    pipeline.fit(x_train_NF, y_train_N)

    print("fit pipeline")
    err = sklearn.metrics.zero_one_loss(y_valid_M, pipeline.predict(x_valid_MF) >= 0.5)
    print(err)


    # print("ARRAYS")
    # print("Raw features: shape %s" % str(x_NF.shape))
    # print(x_NF)
    # print("Transformed feature array phi_NG: shape %s" % str(phi_NG.shape))
    # print(phi_NG)
    #
    # print("NAMES")
    # phi_feat_names = pipeline.named_steps['step1'].get_feature_names()
    # print("Raw features fed into the pipeline:")
    # print(orig_feat_names)
    # print("Transformed features produced by the pipeline:")
    # print(phi_feat_names)

if __name__ == '__main__':
    make_model()