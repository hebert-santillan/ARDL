import numpy as np
import pandas as pd
import statsmodels.api as sm
from pprint import pprint

def lag_grid(num_vars_x, max_lag_y, max_lag_x):

     # Posible lags for y
    lags_y = np.arange(1, max_lag_y + 1)
    
    # Posible lags for x
    lags_x = [np.arange(0, max_lag_x + 1) for i in range(num_vars_x)]
    
    # Number of X combinations
    shapes = [len(lag) for lag in lags_x]
    
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
    return grid

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

    return ([AIC_b, BIC_b])

def ols_model(y, x): 
    """
    Fit an OLS model and return the model along with the SSE (not adjusted for degrees of freedom)
    """
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    y_pred = model.predict(x)
    ehat = y - y_pred
    SSE = np.sum(ehat ** 2)

    return [model, SSE]

def optimal_lag_selection(data, max_lags_y, max_lags_x):
    
    """
    Return the optimal lag combination based on AIC and BIC criteria.
    """
    AIC_list = []
    BIC_list = []
    
    reg_matrix_list = []
    reg_headers_list = []
    SSE_list = []
            
    # Combinations of posible optimal lags        
    combinations = lag_grid(data.shape[1]-1,max_lags_y,max_lags_x)
    
    
    # Convert data structure from DataFrame to matrix, gets matrix shape
    data_matrix = data.to_numpy()
    n = data_matrix.shape [0]  
    m = data_matrix.shape [1]
    
    # header
    headers = data.columns
    
    # lagged matrices
    lagged_matrices = {}
        
    for pos, header in enumerate(headers): 
        # getting data matrix 
        data_values = data_matrix[:,pos]
        data_values = data_values.flatten()
                
        # Lags of the y variable
        if pos == 0:
            # Creating an empty matrix
            lagged_matrix = np.full((n, max_lags_y), np.nan)
            
            # Adding columns with lags from 1 to max_lags_y
            for lag_y in range(max_lags_y):
                lagged_matrix[(lag_y+1):, lag_y] = data_values[:n-(lag_y+1)]
            
        # lags in x variable
        else:
             # Creating an empty matrix
            lagged_matrix = np.full((n, max_lags_x), np.nan)
            
            # Adding columns with lags from 0 to max_lags_x
            for lag_x in range(max_lags_x):
                lagged_matrix[(lag_x+1):, lag_x] = data_values[:n-(lag_x+1)]
            
        # Appending to non-lagged data
        lagged_matrix = np.hstack((data_values.reshape(-1,1), lagged_matrix))
        
        # Saving matrix in dictionary
        lagged_matrices[header] = lagged_matrix
        
    
    for combination in combinations:
        # name of columns
        reg_headers = []
        
        # column number
        n_columns = sum(combination) + m
    
        reg_matrix = np.full((n, n_columns), np.nan)
                
        i = 0
        for pos, lags in enumerate(combination):
            header = headers[pos]
            for lag in range(lags+1):
                reg_matrix[:,i] = lagged_matrices[header][:,lag]   
                i+=1
                if lag == 0:
                    reg_headers.append(header)
                else:
                    reg_headers.append(header+"_lag" + str(lag))
        reg_matrix = reg_matrix[~np.isnan(reg_matrix).any(axis=1)]

        y = reg_matrix[:,0]
        x = reg_matrix[:,1:]

        model, SSE = ols_model(y, x)

        N = reg_matrix.shape[0]
        M = reg_matrix.shape[1]

        AIC, BIC = criterium(N, M, SSE)
        
        AIC_list.append(AIC)
        BIC_list.append(BIC)
        reg_matrix_list.append(reg_matrix)
        reg_headers_list.append(reg_headers)
        SSE_list.append(SSE)

    AIC_opt = AIC_list.index(min(AIC_list))
    BIC_opt = BIC_list.index(min(BIC_list))
    reg_matrix_opt_AIC = reg_matrix_list[AIC_opt]
    reg_matrix_opt_BIC = reg_matrix_list[BIC_opt]
    
    print("Number of models evaluated:" + str(len(combinations)))
    # Print AIC results
    print("########## AIC ##########")
    print("Optimal lags (AIC): " + str([combinations[AIC_opt]]))
    print("SSE: " + str(SSE_list[AIC_opt]))
    print("AIC: " + str(AIC_list[AIC_opt]))
    
    
    df_AIC_opt = pd.DataFrame(reg_matrix_opt_AIC, columns = reg_headers_list[AIC_opt])
    print (ols_model(df_AIC_opt.iloc[:,0], df_AIC_opt.iloc[:,1:])[0].summary())
    
    # Print BIC results
    print("########## BIC ##########")
    print("Optimal lags (BIC): " + str([combinations[BIC_opt]]))
    print("SSE: " + str(SSE_list[AIC_opt]))
    print("BIC: " + str(BIC_list[BIC_opt]))
    
    df_BIC_opt = pd.DataFrame(reg_matrix_opt_BIC, columns = reg_headers_list[BIC_opt])
    print (ols_model(df_BIC_opt.iloc[:,0], df_BIC_opt.iloc[:,1:])[0].summary())
    
    # Returns arrays of optimal lags [AIC, BIC]
    return [combinations[AIC_opt], combinations[BIC_opt]]


import os
filepath = __file__
directory_path = os.path.dirname(__file__)
# Import data sample 1
csv_file_path = "sample.csv"
df_1 = pd.read_csv(os.path.join(directory_path,csv_file_path))


data_1 = df_1.copy()
max_lags_y, max_lags_x = 4,4
optimal_lag_selection(data_1, max_lags_y, max_lags_x)
