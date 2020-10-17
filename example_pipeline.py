'''
Short examples of how to design custom pipelines that
concatenate multiple feature transformations.

See Also
--------
See also the day04 lab on Pipelines in sklearn.

See also the sklearn documentation on pipelines.
'''

import numpy as np
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

class SquaredFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self):
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Args
        ----
        x : 2D array, size F

        Returns
        -------
        feat : 2D array, size N x F
            One feature extracted for each example
        """
        return np.square(x)

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['square_of_%02d' % f for f in range(x.shape[1])]
        return self

class AverageValueFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to the *sum* of all pixels in image
    """

    def __init__(self):
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Returns
        -------
        feat : 2D array, size N x 1
            One feature extracted for each example
        """
        return np.sum(x, axis=1)[:,np.newaxis]

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['avg_of_%s-%s' % (0, x.shape[1])]
        return self

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

    training_BCEs = []
    valid_BCEs = []
    training_ERs = []
    valid_ERs = []

    for i in range(1,40):
        classifier = sklearn.linear_model.LogisticRegression(C=1e6, solver='lbfgs', max_iter=i)
        classifier.fit(x_train_NF, y_train_N)
        yproba1_N_train = classifier.predict(x_train_NF)
        training_BCEs.append(sklearn.metrics.log_loss(y_train_N, yproba1_N_train))
        training_ERs.append(sklearn.metrics.zero_one_loss(y_train_N, yproba1_N_train >= 0.5))

        yproba1_M_valid = classifier.predict(x_valid_MF)
        valid_BCEs.append(sklearn.metrics.log_loss(y_valid_M, yproba1_M_valid))
        valid_ERs.append(sklearn.metrics.zero_one_loss(y_valid_M, yproba1_M_valid >= 0.5))

    # print(training_BCEs)
    # print(training_ERs)
    # print(valid_BCEs)
    # print(valid_ERs)

    fig_handle, axis_handles = plt.subplots(
        nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10,5)
    )

    ax0 = axis_handles[0]
    ax1 = axis_handles[1]
    i_values = np.arange(1, 40)
    ax0.plot(i_values, training_BCEs, 'b.-', label='train binary cross entropy')
    ax0.plot(i_values, valid_BCEs, 'r.-', label='valid binary cross entropy')
    ax0.set_title('Log Loss vs Iteration')
    ax0.set(ylabel='Log Loss', xlabel='max_iters')
    ax0.legend(loc="upper right")

    ax1.plot(i_values, training_ERs, 'b:', label='train err')
    ax1.plot(i_values, valid_ERs, 'r:', label='valid err')
    ax1.set_title('Error Rate vs Iteration')
    ax1.set(ylabel='Error rate', xlabel='max_iters')
    ax1.legend(loc="upper right")

    plt.show()

    # orig_feat_names = ['pixel%02d' % f for f in range(F)]

    # feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
    #         ('orig', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
    #         ('sq', SquaredFeatureExtractor()),
    #         ('av', AverageValueFeatureExtractor()),
    #         ])
    # classifier = sklearn.linear_model.LogisticRegression(C=1.0)
    #
    # pipeline = sklearn.pipeline.Pipeline([
    #     ('step1', feature_tfmr),
    #     ('step2', classifier)
    #     ])
    # pipeline.fit(x_NF, y_N)
    #
    # phi_NG = pipeline.named_steps['step1'].transform(x_NF)
    #
    #
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
    train_i_range()