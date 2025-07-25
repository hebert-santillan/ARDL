import numpy as np
import pandas as pd
import statsmodels.api as sm
import time 
from pprint import pprint

"""
This repository provides a Python implementation of an algorithm for
automatic lag selection in Autoregressive Distributed Lag (ARDL) models.
The tool evaluates all possible combination of lag selection and returns
the optimal lag selection with AIC and BIC. The tool is designed to assist
on research in identifying the optimal lag structure for ARDL models,
facilitating time series analysis and econometric modeling.

1. Create matrix of original values and lagged values:
    y_lag0, y_lag1, ..., y_lag(n), x1_lag0 ...
    
2. Create index of all posible combinations

3. Index lagged matrix and evaluates each combination of lags

4. Print summary of optimal lags

5. Return array of optimal lag selection

"""



# Clear memory
import gc
gc.collect()

def lag_grid(num_vars_x, max_lag_y, max_lag_x):
    """
    Create a matrix of posible lags, returns a matrix wiht boolean values to index of lagged matrix [True, True, ...]
    """
     # Posible lags for y
    lags_y = np.arange(1, max_lag_y + 1)
    
    # Posible lags for x
    lags_x = [np.arange(0, max_lag_x + 1) for i in range(num_vars_x)]
    
    # Number of X combinations
    
    # Create lag grids for X variables
    grids = np.meshgrid(*lags_x, indexing='ij')
    
    # Flatten and combine
    X_combinations = np.stack([g.flatten() for g in grids], axis=-1)
    
    # Stack results
    result = []
    for y_lag in lags_y:
        y_column = np.full((X_combinations.shape[0], 1), y_lag)
        combo = np.hstack([y_column, X_combinations])
        result.append(combo)

    grid = np.vstack(result)


    max_lag = max(max_lag_x, max_lag_y) + 1

    # Matrix of posible lags
    elements_y = np.array([[True]*lag + [False]*(max_lag_y+1-lag) for lag in range(1,max_lag_y+1+1)])
    elements_x = np.array([[True]*lag + [False]*(max_lag_x+1-lag) for lag in range(1,max_lag_x+1+1)])

    
    y_grid = grid[:,0]
    x_grid = grid[:,1:]
    
    combinations_y = elements_y[y_grid.reshape (1,-1)[0]].reshape(grid.shape[0],-1)        
    combinations_x = elements_x[x_grid.reshape (1,-1)[0]].reshape(grid.shape[0],-1)        
    
    # All posible combinations, to boolean index of lagged matrix [True, True, ...]
    combinations = np.concatenate([
        combinations_y,
        combinations_x]
        , axis = 1)

    
    return combinations

def criterium(N, M, SSE): 
    """
    Evaluate a model according to the Akaike and Bayesian information criteria
    """
    log_lik = (-N/2) * (1 + np.log(2 * np.pi) + np.log(SSE)/N)

    AIC = ((-2 * log_lik)/N) + (2*M/N)
    AIC_a = (2 * M) - (2*log_lik)
    AIC_b = np.log(SSE / N)+(2 * M/N)

    BIC = ((-2 * log_lik)/N) + (M * np.log(N) / N)
    BIC_a = (-2 * log_lik) + (M * np.log(N))
    BIC_b = np.log(SSE / N)+(M * np.log(N))

    # Set to third option
    return ([AIC_b, BIC_b])

def ols_model_sse(y, x): 
    """
    Fit an OLS model and return the SSE (not adjusted for degrees of freedom)
    """
    w = np.linalg.inv(x.T @ x) @ x.T @ y

    y_pred = x @ w
    
    sse = np.sum((y - y_pred) ** 2)
    
    return (sse)
   
def ols_model(y, x): 
    """
    Fit an OLS model and return the model (not adjusted for degrees of freedom)
    """
    model = sm.OLS(y, x).fit()

    y_pred = model.predict(x)
    ehat = y - y_pred
    SSE = np.sum(ehat ** 2)

    return (model)

def optimal_lag_selection(data, max_lags_y, max_lags_x):    
    ### 1 ###
    
    # Get headers
    td1 = time.time() # Chronometer start

    headers_list = data.columns
    
    # Convert data structure from DataFrame to matrix, gets matrix shape
    data_matrix = data.to_numpy()
    
    # Divide data into y and x matrices
    data_matrix_x = data_matrix[:,1:]
    data_matrix_y = data_matrix[:,:1]
    

    # Headers lagged y
    lagged_headers_y = [
    "{}_lag{}".format(headers_list[0], lag) 
    for lag in range(0, max_lags_y + 1) 
    
    ]
    
    # Lagging y matrix
    n, m = data_matrix_y.shape
    lagged_list_y = [
        np.vstack((
            np.full((lag, m), np.nan),
            data_matrix_y[:-lag]
        )) if lag > 0 else data_matrix_y  # lag = 0 = matriz original
        for lag in range(0, max_lags_y + 1)
    ]
    
    # Headers lagged x
    lagged_headers_x = [
    "{}_lag{}".format(i, lag) 
    for i in headers_list[1:]
    for lag in range(0, max_lags_x + 1)
    ]

    n, m = data_matrix_x.shape
    
    # Lagging x matrix
    lagged_list_x = [
        [
            np.vstack((
                np.full((lag, 1), np.nan),
                data_matrix_x[:-lag, i].reshape(-1, 1)
            )) if lag > 0 else data_matrix_x[:, i].reshape(-1, 1)
            for lag in range(0, max_lags_x + 1)  # lag 0 first
        ]
        for i in range(m)
    ]
    
    # Combine into a single matrix (lag 0 -> lag N)
    lagged_matrix_x = np.hstack([
        np.hstack(n) for n in lagged_list_x
    ])

    lagged_matrix_y = np.hstack(lagged_list_y)
  
    lagged_matrix = np.hstack((lagged_matrix_y, lagged_matrix_x))
    lagged_headers = lagged_headers_y + lagged_headers_x 
    
    ### 2 ###
    # Create combinations of posible optimal lags        
    combinations = lag_grid (data.shape[1]-1,max_lags_y,max_lags_x)
    print("Number of models evaluated: " + str(len(combinations)))
    
    ### 3 ###
    
    AIC_list = []
    BIC_list = []
    
    # Evaluate each combination of posible optimal lags
    td2 = time.time() # Chronometer end 
    print ("---- Chronometer (for calculating combinations): " + str(td2-td1)+" ----")
    
    ts1 = 0 # Chronometer start 
    tr1 = 0 # Chronometer start 

    for position, combination in enumerate (combinations):
        # Index columns of lagged values
        ts2 = time.time()

        reg_matrix = lagged_matrix[:,combination]
        reg_matrix = reg_matrix[~np.isnan(reg_matrix).any(axis=1)]
        
        ts3 = time.time()
        # count
        if position % 100000 == 0:
            print ("---- " + str(int(position)) + " models evaluated ----")
        
        tr2 = time.time()

        # Calculate SSE
        sse = ols_model_sse(
            reg_matrix[:,0],  # Y value
            reg_matrix[:,1:]) # X values
        N = reg_matrix.shape[0]
        M = reg_matrix.shape[1]
    
        # Evaluate based on the AIC and BIC
        AIC, BIC = criterium(N, M, sse)
        
        AIC_list.append(AIC)|
        BIC_list.append(BIC)

        tr3 = time.time()

        tr1 += tr3 - tr2
        ts1 += ts3 - ts2

    print ("---- All " + str(len(combinations)) + " models have been evaluated ----")
    print ("---- Chronometer (for indexing models): " + str(ts1)+" ----\n")
    print ("---- Chronometer (for evaluating models): " + str(tr1)+" ----\n")

    ### 4 ###
    
    AIC_opt = AIC_list.index(min(AIC_list))
    BIC_opt = BIC_list.index(min(BIC_list))
    
    combination_AIC_int = [int(x) for x in combinations[AIC_opt]]
    combination_AIC_lst = [sum(combination_AIC_int[:max_lags_y+1])-1] + [sum (x)-1 for x in np.array(combination_AIC_int[max_lags_y+1:]).reshape(-1,max_lags_x+1)]

    combination_BIC_int = [int(x) for x in combinations[BIC_opt]]
    combination_BIC_lst = [sum(combination_BIC_int[:max_lags_y+1])-1] + [sum (x)-1 for x in np.array(combination_BIC_int[max_lags_y+1:]).reshape(-1,max_lags_x+1)]
        
    print("########## AIC ##########")
    print("Optimal lags (AIC): " + str(combination_AIC_lst))
    
    reg_matrix_AIC = lagged_matrix[:,combinations[AIC_opt]]
    reg_matrix_AIC = reg_matrix_AIC[~np.isnan(reg_matrix_AIC).any(axis=1)]
    
    columns = np.array(lagged_headers)[combinations[AIC_opt]]
    df_AIC_opt = pd.DataFrame(
        reg_matrix_AIC, columns=columns)

    # Calculate Full Model
    model_AIC = ols_model(
        df_AIC_opt.iloc[:,0],  # Y value
        df_AIC_opt.iloc[:,1:]) # X values
    N = df_AIC_opt.shape[0]
    M = df_AIC_opt.shape[1]
    
    print (model_AIC.summary())
   
    
    # Print BIC results
    print("\n########## BIC ##########")
    print("Optimal lags (BIC): " + str(combination_BIC_lst))
    
    reg_matrix_BIC = lagged_matrix[:,combinations[BIC_opt]]
    reg_matrix_BIC = reg_matrix_BIC[~np.isnan(reg_matrix_BIC).any(axis=1)]
    
    columns = np.array(lagged_headers)[combinations[BIC_opt]]
    df_BIC_opt = pd.DataFrame(
        reg_matrix_BIC, columns=columns)

    # Calculate Full Model
    model_BIC = ols_model(
        df_BIC_opt.iloc[:,0],  # Y value
        df_BIC_opt.iloc[:,1:]) # X values
    N = df_BIC_opt.shape[0]
    M = df_BIC_opt.shape[1]
    
    print (model_BIC.summary())
    
    
    ### 5 ###
    
    return [combination_AIC_lst, combination_BIC_lst]

### Example ###
import os
directory_path = os.path.abspath('')
url = "https://github.com/marco-amh/codes/raw/refs/heads/master/Data.xlsx"
df = pd.read_excel(url, sheet_name = 'Demanda_dinero', index_col = 0)

t1 = time.time()

data_2 = df.copy()
max_lags_y, max_lags_x = 5, 5 
results = optimal_lag_selection(data_2, max_lags_y, max_lags_x)

t2 = time.time()

print ("\n---- Chronometer (Total time): " + str(t2-t1)+" ----")
print (results)
