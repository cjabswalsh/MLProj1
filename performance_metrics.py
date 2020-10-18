import numpy as np
import pandas as pd
import sklearn.metrics

def calc_mean_squared_error(y_N, yhat_N):
    sum = 0.0
    for i in range (len(y_N)):
        sum += np.square(y_N[i] - yhat_N[i])
    return float((1/(len(y_N))) * sum)


def calc_mean_absolute_error(y_N, yhat_N):
    sum = 0.0
    for i in range (len(y_N)):
        sum += np.abs(y_N[i] - yhat_N[i])
    return float((1/(len(y_N)) * sum))
    
def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    yhat_N = np.asarray(yhat_N, dtype=np.int32)
    N, = ytrue_N.shape
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(N):
        if(ytrue_N[i] == 1):
            if(yhat_N[i] == 1):
                TP += 1
            else: 
                FN += 1
        else: 
            if(yhat_N[i] == 0):
                TN += 1
            else: FP += 1
                
    return TP, TN, FP, FN


def calc_ACC(ytrue_N, yhat_N): 
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    return (TP + TN)/(len(ytrue_N) + .000000000000000001)

def calc_TPR(ytrue_N, yhat_N):
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    return TP/(TP + FN + .000000000000000001)


def calc_TNR(ytrue_N, yhat_N):
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    return TN/(TN + FP + .000000000000000001)



def calc_PPV(ytrue_N, yhat_N):
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    return TP/(TP + FP + .000000000000000001)


def calc_NPV(ytrue_N, yhat_N):
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
    return TN/(TN + FN + .000000000000000001) 
    
import pandas as pd
import sklearn.metrics

def calc_confusion_matrix_for_probas_and_threshold(ytrue_N, yproba1_N, thr):
    
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    yproba1_N = np.asarray(yproba1_N, dtype=np.float64)
    yhat_N = np.asarray(yproba1_N >= thr, dtype=np.int32)

    cm = sklearn.metrics.confusion_matrix(ytrue_N, yhat_N)
    cm_df = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
    cm_df.columns.name = 'Predicted'
    cm_df.index.name = 'True'
    return cm_df