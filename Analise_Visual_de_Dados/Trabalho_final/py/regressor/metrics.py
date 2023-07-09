from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from scipy.stats import pearsonr
import numpy as np

def pearson_score(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr

def adj_r2_score(estimator, X, y_true):
    n, p = X.shape
    pred = estimator.predict(X)
    return 1 - ((1 - r2_score(y_true, pred)) * (n - 1))/(n-p-1)
