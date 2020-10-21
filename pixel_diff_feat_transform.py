import numpy as np

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from baseline_prob2 import train_cv_model

class RowAverageExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self, row_size):
        self.row_size = row_size
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
        N, F = x.shape
        print(N, F)
        assert(F % self.row_size == 0)
        num_pixel_rows = int(F / self.row_size)

        pixel_rows = [np.split(a, num_pixel_rows) for a in x]
        return np.sum(pixel_rows, axis=2)


    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['row_sum_%02d' % f for f in range(28)]
        return self

if __name__ == '__main__':
    feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
        ('orig', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
        ('sq', RowAverageExtractor(row_size=28)),
    ])
    classifier = sklearn.linear_model.LogisticRegression(C=1.0)
    rescaler = sklearn.preprocessing.MinMaxScaler()

    pipeline = sklearn.pipeline.Pipeline([
        ('rescaler', rescaler),
        ('transform', feature_tfmr),
        ('classifier', classifier)
    ])
    train_cv_model(pipeline)