
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time 
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

def lag_grid(num_vars, max_lag_y, max_lag_x):
       
    
     # Posible lags for y
    lags_y = np.arange(1, max_lag_y + 1)
    
    # Posible lags for x
    lags_x = [np.arange(0, max_lag_x + 1) for i in range(num_vars)]
    
    # Number of X combinations
    
    # Create lag grids for X variables
    grids = np.meshgrid(*lags_x, indexing='ij')
    
    
    # Flatten and combine
    X_combinations = np.stack([g.flatten() for g in grids], axis=-1)
    print (X_combinations)
    
    # Stack results
    result = []
    for y_lag in lags_y:
        y_column = np.full((X_combinations.shape[0], 1), y_lag)
        combo = np.hstack([y_column, X_combinations])
        result.append(combo)
    
    grid = np.vstack(result)
    return grid


# lag_grid (4,3,3)


def lag_grid2 (num_vars, max_lag_y, max_lag_x):
    
    grids = [np.diag([1]*(n) + [0]*(max_lag_x-n)) for n in range(1,max_lag_x+1) for m in range(num_vars)]
    
    
    
    
    
    
    
    
    
    
    # for n in range (num_vars):
    # # # Create lag grids for X variables
    #     grids = np.meshgrid(*lags_list, indexing='ij')
        
        
    # #   # Posible lags for y
    # lags_y = np.arange(1, max_lag_y + 1)
    
    
    # # Posible lags for x
    # lags_x = [np.arange(0, max_lag_x + 1) for i in range(num_vars)]
    
    # # Number of X combinations
    
    
    
    # # Flatten and combine
    # X_combinations = np.stack([g.flatten() for g in grids], axis=-1)
    # print (X_combinations)
    
    # # Stack results
    # result = []
    # for y_lag in lags_y:
    #     y_column = np.full((X_combinations.shape[0], 1), y_lag)
    #     combo = np.hstack([y_column, X_combinations])
    #     result.append(combo)
    
    # grid = np.vstack(result)
    # return grid
    return grids


pprint (lag_grid2(2,3,3))

