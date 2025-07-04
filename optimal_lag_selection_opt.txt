import numpy as np
import pandas as pd
import statsmodels.api as sm
import time 
from pprint import pprint


import gc
gc.collect()

def lag_grid(num_vars_x, max_lag_y, max_lag_x):

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
    # print ("### grid ###")
    grid = np.vstack(result)
    # print (pd.DataFrame(grid))

    max_lag = max(max_lag_x, max_lag_y) + 1

    # print ("### Elements ###")    
    elements_y = np.array([[True]*lag + [False]*(max_lag_y+1-lag) for lag in range(1,max_lag_y+1+1)])
    elements_x = np.array([[True]*lag + [False]*(max_lag_x+1-lag) for lag in range(1,max_lag_x+1+1)])


    # print (pd.DataFrame(elements_x))
    # print (pd.DataFrame(elements_y))
    
    y_grid = grid[:,0]
    x_grid = grid[:,1:]
    
    combinations_y = elements_y[y_grid.reshape (1,-1)[0]].reshape(grid.shape[0],-1)        
    combinations_x = elements_x[x_grid.reshape (1,-1)[0]].reshape(grid.shape[0],-1)        
    
    # print (pd.DataFrame(combinations_y))
    # print (pd.DataFrame(combinations_x))
    
    combinations = np.concatenate([
        combinations_y,
        combinations_x]
        , axis = 1)
    
    # print (pd.DataFrame(combinations))
    
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

    return ([AIC_b, BIC_b])

def ols_model_sse(y, x): 
    w = np.linalg.inv(x.T @ x) @ x.T @ y

    # Step 2: Predict y using the model
    y_pred = x @ w
    
    # Step 3: Calculate SSE
    sse = np.sum((y - y_pred) ** 2)
    
    return (sse)

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
    # Get headers
    headers_list = data.columns
    
    # Convert data structure from DataFrame to matrix, gets matrix shape
    data_matrix = data.to_numpy()
    
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
    
    # Lagging x matrix
    n, m = data_matrix_x.shape
    lagged_list_x = [
        np.vstack((
            np.full((lag, m), np.nan),
            data_matrix_x[:-lag]
        )) if lag > 0 else data_matrix_x  # lag = 0 = matriz original
        for lag in range(0, max_lags_x + 1)
    ]
    
    lagged_matrix_x = np.hstack(lagged_list_x)
    lagged_matrix_y = np.hstack(lagged_list_y)
  
    lagged_matrix = np.hstack((lagged_matrix_y, lagged_matrix_x))
    lagged_headers = lagged_headers_y + lagged_headers_x 
            
    # Combinations of posible optimal lags        
    combinations = lag_grid (data.shape[1]-1,max_lags_y,max_lags_x)
    
    # print ("Models evaluated: {}".format(combinations.shape[0]))
    ts1 = time.time()
    
    for position, combination in enumerate (combinations):
        results = lagged_matrix[:,combination][~np.isnan(lagged_matrix).any(axis=1)]
        
        sse = ols_model_sse(results[:,0], results[:,1:])
        N = results.shape[0]
        M = results.shape[1]
                
        if position == 0:
            opt_AIC, opt_BIC = criterium(N, M, sse)
            best_AIC = 0
            best_BIC = 0
        else:
            opt_AIC_n, opt_BIC_n = criterium(N, M, sse)
            if opt_AIC_n < opt_AIC:
                opt_AIC = opt_AIC_n
                best_AIC = position
            
            if opt_BIC_n < opt_BIC:
                opt_BIC = opt_BIC_n
                best_BIC = position
    
        combination_a = combination
        combination_a_int = [int(x) for x in combination_a]
        
        x = [sum(combination_a_int[:max_lags_y+1])-1] + [sum (x)-1 for x in np.array(combination_a_int[max_lags_y+1:]).reshape(-1,max_lags_x+1)]
        if x == [1,0,0,0]:
            a, b = criterium(N, M, sse)
            print (b)
        
    ### Getting best model with headers
    ### AIC ###
    combination_AIC = combinations[best_AIC]
    combination_AIC_int = [int(x) for x in combination_AIC]

    print ([sum(combination_AIC_int[:max_lags_y])] +
           [sum (x) for x in np.array(combination_AIC_int[max_lags_y+1:]).reshape(-1,max_lags_x+1)])

    combination_BIC = combinations[best_BIC]
    combination_BIC_int = [int(x) for x in combination_BIC]

    print ([sum(combination_BIC_int[:max_lags_y+1])] +
           [sum (x) for x in np.array(combination_BIC_int[max_lags_y+1:]).reshape(-1,max_lags_x+1)])
        
    
    
    # results = lagged_matrix[:,combinations[best_AIC]][~np.isnan(lagged_matrix).any(axis=1)]
    # pd.DataFrame(results[:,:(max_lags_y-1)],columns = np.array(lagged_headers)[combinations[best_AIC]][:(max_lags_y-1)])
    
    ### BIC ###
    print (combinations[best_BIC])
    # pd.DataFrame(results[:,max_lags_y:],columns = np.array(lagged_headers)[combination][max_lags_y:])
            
    
        
            
            
        
        # if position == 200:
                
            # identity_matrix = np.zeros((n,n))
            
            # np.fill_diagonal (identity_matrix, combination)
            
            # C = lagged_matrix @ identity_matrix
            # print (results)
                      
            # print (pd.DataFrame(lagged_headers).iloc[combination,])
            # print (pd.DataFrame(results))
            
        
            # print (pd.DataFrame(results[:,:(max_lags_y-1)],columns = np.array(lagged_headers)[combination][:(max_lags_y-1)]))
            # print (pd.DataFrame(results[:,max_lags_y:],columns = np.array(lagged_headers)[combination][max_lags_y:]))
                                
            
        
        

        
    ts2 = time.time()
    print ("Tiempo de t2-t1: " + str(ts2-ts1))
    
    # [position*max_lags_x + lag for lag in range(0,lags+1) for lags in combination for position, combination in enumerate(combinations)]    
    # [print (str(lag_x*(m+1)+position) + "-"+str(position)+ "-"+str(combination))
    #   for combination in combinations
    #   for position, lags_x in enumerate(combination[1:])
    #   for lag_x in range(lags_x+1)]
                        
            
            
            
    
    
    # y_data_matrix = data_matrix[:,0]
    
    # y_lagged_matrix = np.full_like(y_data_matrix, np.nan, dtype=float)
    # y_lagged_matrix[lag:] = y_data_matrix[:-lag]

    # np.full_like(matrix, np.nan, dtype=float)

    
    # """
    # Return the optimal lag combination based on AIC and BIC criteria.
    # """
    # AIC_list = []
    # BIC_list = []
    
    # reg_matrix_list = []
    # reg_headers_list = []
    # SSE_list = []
    
    # n = data_matrix.shape [0]  
    # m = data_matrix.shape [1]
    
    # # header
    # headers = data.columns
    
    
    
    
    # for pos, header in enumerate(headers): 
    #     # getting data matrix 
    #     data_values = data_matrix[:,pos]
    #     data_values = data_values.flatten()
                
    #     # Lags of the y variable
    #     if pos == 0:
    #         # Creating an empty matrix
    #         lagged_matrix = np.full((n, max_lags_y), np.nan)
            
    #         # Adding columns with lags from 1 to max_lags_y
    #         for lag_y in range(max_lags_y):
    #             lagged_matrix[(lag_y+1):, lag_y] = data_values[:n-(lag_y+1)]
            
    #     # lags in x variable
    #     else:
    #          # Creating an empty matrix
    #         lagged_matrix = np.full((n, max_lags_x), np.nan)
            
    #         # Adding columns with lags from 0 to max_lags_x
    #         for lag_x in range(max_lags_x):
    #             lagged_matrix[(lag_x+1):, lag_x] = data_values[:n-(lag_x+1)]
            
    #     # Appending to non-lagged data
    #     lagged_matrix = np.hstack((data_values.reshape(-1,1), lagged_matrix))

    # print (lagged_matrix)
    
        
        
    #     # # name of columns
    #     # reg_headers = []
        
    #     # # column number
    #     # n_columns = sum(combination) + m
    
    #     # reg_matrix = np.full((n, n_columns), np.nan)
                
    #     # i = 0
    #     # for pos, lags in enumerate(combination):
    #     #     header = headers[pos]
    #     #     for lag in range(lags+1):
    #     #         reg_matrix[:,i] = lagged_matrices[header][:,lag]   
    #     #         i+=1
    #     #         if lag == 0:
    #     #             reg_headers.append(header)
    #     #         else:
    #     #             reg_headers.append(header+"_lag" + str(lag))
    #     # reg_matrix = reg_matrix[~np.isnan(reg_matrix).any(axis=1)]
        
        
        
        
        
    # #     y = reg_matrix[:,0]
    # #     x = reg_matrix[:,1:]

    # #     model, SSE = ols_model(y, x)

    # #     N = reg_matrix.shape[0]
    # #     M = reg_matrix.shape[1]

    # #     AIC, BIC = criterium(N, M, SSE)
        
    # #     AIC_list.append(AIC)
    # #     BIC_list.append(BIC)
    # #     reg_matrix_list.append(reg_matrix)
    # #     reg_headers_list.append(reg_headers)
    # #     SSE_list.append(SSE)

    # # AIC_opt = AIC_list.index(min(AIC_list))
    # # BIC_opt = BIC_list.index(min(BIC_list))
    # # reg_matrix_opt_AIC = reg_matrix_list[AIC_opt]
    # # reg_matrix_opt_BIC = reg_matrix_list[BIC_opt]
    
    # # print("Number of models evaluated:" + str(len(combinations)))
    # # # Print AIC results
    # # print("########## AIC ##########")
    # # print("Optimal lags (AIC): " + str([combinations[AIC_opt]]))
    # # print("SSE: " + str(SSE_list[AIC_opt]))
    # # print("AIC: " + str(AIC_list[AIC_opt]))
    
    
    # # df_AIC_opt = pd.DataFrame(reg_matrix_opt_AIC, columns = reg_headers_list[AIC_opt])
    # # print (ols_model(df_AIC_opt.iloc[:,0], df_AIC_opt.iloc[:,1:])[0].summary())
    
    # # # Print BIC results
    # # print("########## BIC ##########")
    # # print("Optimal lags (BIC): " + str([combinations[BIC_opt]]))
    # # print("SSE: " + str(SSE_list[AIC_opt]))
    # # print("BIC: " + str(BIC_list[BIC_opt]))
    
    # # df_BIC_opt = pd.DataFrame(reg_matrix_opt_BIC, columns = reg_headers_list[BIC_opt])
    # # print (ols_model(df_BIC_opt.iloc[:,0], df_BIC_opt.iloc[:,1:])[0].summary())
    
    # # # Returns arrays of optimal lags [AIC, BIC]
    # # return [combinations[AIC_opt], combinations[BIC_opt]]

# ### Example ###
# import os
# directory_path = os.path.abspath('')
# # Import data sample 1
# csv_file_path = "sample.csv"
# df_1 = pd.read_csv(os.path.join(directory_path,csv_file_path))


# data_1 = df_1.copy()
# max_lags_y, max_lags_x = 4,4
# optimal_lag_selection(data_1, max_lags_y, max_lags_x)


### Example ###
import os
directory_path = os.path.abspath('')
# Import data sample 1
csv_file_path = "sample.csv"
df_1 = pd.read_csv(os.path.join(directory_path,csv_file_path))


data_1 = df_1.copy()
max_lags_y, max_lags_x = 4,4
optimal_lag_selection(data_1, max_lags_y, max_lags_x)

# # ### Data Marco ###
# import os
# directory_path = os.path.abspath('')
# url = "https://github.com/marco-amh/codes/raw/refs/heads/master/Data.xlsx"
# df = pd.read_excel(url, sheet_name = 'Demanda_dinero', index_col = 0)

# t1 = time.time()

# data_2 = df.copy()
# max_lags_y, max_lags_x = 8, 8
# optimal_lag_selection(data_2, max_lags_y, max_lags_x)

# t2 = time.time()

# print ("Tiempo de t2-t1: " + str(t2-t1))


