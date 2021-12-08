import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from math import sqrt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_residuals(df, y, yhat):
    '''
    Takes in y and yhat to return a plot of residuals
    '''
    plt.figure(figsize=(10,10))
    plt.axhline(color = 'green')
    plt.scatter(df[y], df[yhat], data = df)
    plt.xlabel(f'{y}')
    plt.ylabel(r'$\hat{y}-y$')
    plt.title('Model Residuals')
    return plt.show()

def regression_errors(y, yhat):
    '''
    Takes y and yhat as arguments and returns the Sum of Squared Errors, 
    Explained Sum of Squares, Total Sum of Squares, Mean Squared Error
    and Root Mean Squared Error
    '''
    # sum of squares error
    SSE = mean_squared_error(y, yhat)*len(y)
    # explained sum of squares
    ESS = sum((yhat - y.mean())**2)
    # mean squared error
    MSE = mean_squared_error(y, yhat)
    # root mean squared error
    RMSE = sqrt(mean_squared_error(y, yhat))
    return SSE, ESS, MSE, RMSE

def baseline_mean_errors(y):
    '''
    Takes y as an argument and returns the baseline Sum of Squared Errors,
    Total Sum of Squares, Mean Squared Error, and Root Mean Squared Error.
    '''
    # SSE baseline
    SSE_baseline = mean_squared_error(y, [[y.mean()]])* len(y)
    # MSE baseline
    MSE_baseline = mean_squared_error(y, y.mean())
    # RMSE baseline
    RMSE_baseline = sqrt(mean_squared_error(y, y.mean()))
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    '''
    Takes y and yhat as arguments, returning True if the model performs better 
    than the baseline. Evaluates the model's Sum of Squared Errors, 
    Total Sum of Squares, Mean Squared Error, and Root Mean Squared Error 
    against those of the baseline.
    '''
    SSE, ESS, MSE, RMSE = regression_errors(y, yhat)
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    if [SSE, MSE, RMSE] < [SSE_baseline, MSE_baseline, RMSE_baseline]:
        return True