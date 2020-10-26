import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product
# from baseline_prob2 import train_static_val, train_cv_model
from scipy import ndimage as img

class NeighborExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self, row_size):
        self.row_size = row_size
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]
        
    def transform(self, x, y=None):
        print("transforming")
        N,F = x.shape
            # new_image = scipy.ndimage.gaussian_filter(new_image, sigma=1)
            # image = new_image.flatten()
        for i in range(N):
            # print("image ", i)
            for j in range(F):
                neighbors = 0
                if j <783: neighbors+= x[i,j+1]
                if j > 1: neighbors+=x[i,j-1]
                if j <755: neighbors+=x[i,j+28]
                if j > 28: neighbors+=x[i,j-28]
                x[i,j] += neighbors/8
                if x[i,j] > 1: x[i,j] = 1
        return x
        
    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['row_sum_%02d' % f for f in range(28)]
        return self

class QuantExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self, row_size):
        self.row_size = row_size
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        return x >= 0.2

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['row_sum_%02d' % f for f in range(28)]
        return self

class NdImageExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self, row_size):
        self.row_size = row_size
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        final = []
        for image in x:
            image.reshape(28, 28)
            open_square = img.binary_opening(image)
            eroded_square = img.binary_erosion(image)
            reconstruction = img.binary_propagation(eroded_square, mask=image)
            final.append(reconstruction.flatten())
        return final

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['row_sum_%02d' % f for f in range(28)]
        return self

class SobelExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self, row_size):
        self.row_size = row_size
        pass

    def get_feature_names(self):
        return [a for a in self.feature_names]

    def transform(self, x, y=None):
        final = []
        for image in x:
            image = image.reshape(28, 28)
            sx = img.sobel(image, axis=0, mode='nearest')
            sy = img.sobel(image, axis=1, mode='nearest')
            sob = np.hypot(sx, sy)

            final.append(sob.flatten())
        return final

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        self.feature_names = ['row_sum_%02d' % f for f in range(28)]
        return self
 
 
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
 
def test_C_vals(x_train_NF, y_train_N, x_valid_MF, y_valid_M, classifier):
    C_grid = np.logspace(-3, 1, 15)

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
    plt.legend(loc="bottom right")


    plt.show()

    print("Best C value is:", C_grid[np.argmin(valid_ERs)])

def train_static_val(estimator, thr = 0.5):
    
    x_train_NF, y_train_N, x_valid_MF, y_valid_M = get_std_data()
    N, = y_train_N.shape
    M, = y_valid_M.shape
    estimator.fit(x_train_NF, y_train_N)
    err_train = sklearn.metrics.zero_one_loss(y_train_N, estimator.predict(x_train_NF) >= thr)
    err_valid = sklearn.metrics.zero_one_loss(y_valid_M, estimator.predict(x_valid_MF) >= thr)
    yproba1_train_N = estimator.predict_proba(x_train_NF)[:,1]
    yproba1_valid_M = estimator.predict_proba(x_valid_MF)[:,1]
    
    yhat_M_valid = estimator.predict(x_valid_MF) >= 0.5
    
    print(y_valid_M[32])
    print(yhat_M_valid[32])
    
    FNs = []
    FPs = []
    
    y_valid_M_bools = y_valid_M >= 0.5
    
    for i in range(len(y_valid_M)):
        if y_valid_M_bools[i] == True and yhat_M_valid[i] == False:
            FNs.append(i+9000)
        elif y_valid_M_bools[i] == False and yhat_M_valid[i] == True:
            FPs.append(i+9000)
    print("FNs:", FNs)
    print("FPs:", FPs)
    return y_valid_M, yproba1_valid_M
    
def baseline_pipeline():
    return sklearn.linear_model.LogisticRegression(C=0.1, solver = 'lbfgs', max_iter = 1000)
   
def basic_pipeline():
    feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
        ('orig', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
        ('simplify', QuantExtractor(row_size=28)),
        # ('neighbors', NeighborExtractor(row_size=28)),
        # ('image_manip', NdImageExtractor(row_size=28)),
        # ('Sobel', SobelExtractor(row_size=28))
    ])
    classifier = sklearn.linear_model.LogisticRegression(C=0.1, solver = 'lbfgs', max_iter = 1000)
    rescaler = sklearn.preprocessing.MinMaxScaler()

    pipeline = sklearn.pipeline.Pipeline([
        ('rescaler', rescaler),
        ('transform', feature_tfmr),
        ('classifier', classifier)
    ])
    return pipeline

def complex_pipeline():
    feature_tfmr = sklearn.pipeline.FeatureUnion(transformer_list=[
        ('orig', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
        ('simplify', QuantExtractor(row_size=28)),
        # ('neighbors', NeighborExtractor(row_size=28)),
        ('image_manip', NdImageExtractor(row_size=28)),
        ('Sobel', SobelExtractor(row_size=28))
    ])
    classifier = sklearn.linear_model.LogisticRegression(C=0.1, solver = 'lbfgs', max_iter = 1000)
    rescaler = sklearn.preprocessing.MinMaxScaler()

    pipeline = sklearn.pipeline.Pipeline([
        ('rescaler', rescaler),
        ('transform', feature_tfmr),
        ('classifier', classifier)
    ])
    return pipeline
    
if __name__ == '__main__':
    baseline_pipeline = baseline_pipeline()
    basic_pipeline = basic_pipeline()
    complex_pipeline = complex_pipeline()
    # print("running baseline")
    # base_y_valid_M, base_yproba1_valid_M = train_static_val(baseline_pipeline, 0.5)
    print("running basic")
    basic_y_valid_M, basic_yproba1_valid_M = train_static_val(basic_pipeline, 0.5)
    # print("running complex")
    # complex_y_valid_M, complex_yproba1_valid_M = train_static_val(complex_pipeline, 0.5)
    
    # base_fpr, base_tpr, ignore_this = sklearn.metrics.roc_curve(base_y_valid_M, base_yproba1_valid_M)
    # basic_fpr, basic_tpr, ignore_this = sklearn.metrics.roc_curve(basic_y_valid_M, basic_yproba1_valid_M)
    # complex_fpr, complex_tpr, ignore_this = sklearn.metrics.roc_curve(complex_y_valid_M, complex_yproba1_valid_M)

    # plt.figure(figsize=(5,5))
    # plt.plot(base_fpr, base_tpr, 'g.-', label='Experiment 0: Baseline')
    # plt.plot(basic_fpr, basic_tpr, 'r.-', label='Experiment 1: Basic Transforms')
    # plt.plot(complex_fpr, complex_tpr, 'b.-', label='Experiment 2: Complex Transforms')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.legend(loc="lower right")
    # plt.show()