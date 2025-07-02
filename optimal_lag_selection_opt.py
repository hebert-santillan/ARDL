import numpy as np
import pandas as pd
import statsmodels.api as sm
import time 
import itertools
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed


# # def lag_grid (num_vars_x, max_lag_y, max_lag_x):
    
#     max_lag = max(max_lag_x, max_lag_y)
    
#       # Posible lags for y
#     lags_y = elements[1:]
#     # lags_y = np.arange(1, max_lag_y + 1)
    
#     # Posible lags for x
#     lags_x = [[elements[i] for i in np.arange(1, max_lag_x + 1)] for i in range(num_vars_x)]
    
#     grid = [list(p) for p in itertools.product(elements, repeat=num_vars_x)]
    
#     combinations = [np.array (row).flatten() for row in grid]
#     # # combinations = [itertools.chain(row) for row in grid]
#     # # combinations = [list(row) for row in grid]
#     # combinations = [list(itertools.chain.from_iterable(row)) for row in itertools.product(elements, repeat=num_vars_x)]
    

#     return (combinations)

def lag_grid2(num_vars_x, max_lag_y, max_lag_x):
    max_lag_x += 1
    max_lag_y += 1

    max_lag = max(max_lag_x, max_lag_y)

    print ("### Elements ###")    
    elements = np.array([[True]*lag + [False]*(max_lag-lag) for lag in range(1,max_lag+1)])
    print (elements)
    
    if num_vars_x == 1:
        grid_np = elements
    else:
        # product_arrays = [elements for _ in range(num_vars_x)]
        
        indices_range = np.arange(max_lag_x)
        
        meshgrid_indices = np.array(np.meshgrid(*[indices_range for _ in range(num_vars_x)], indexing='ij'))
        
        grid_indices_flat = meshgrid_indices.reshape(num_vars_x, -1).T
        
        grid_np = elements[grid_indices_flat]
        print (grid_np)
    
    combinations_x = grid_np.reshape(-1, num_vars_x * max_lag)
    # for lag_y in range (max_lag_y)
    # combinations = np.meshgrid (combinations_x,max_lag_y)
    
    return (combinations_x)

def lag_grid3(num_vars_x, max_lag_y, max_lag_x):
    max_lag_x += 1
    max_lag_y += 1

    max_lag = max(max_lag_x, max_lag_y)

    print ("### Elements ###")    
    elements = np.array([[True]*lag + [False]*(max_lag-lag) for lag in range(1,max_lag+1)])
    print (elements)
    
    if num_vars_x == 1:
        grid_np = elements
    else:
        # product_arrays = [elements for _ in range(num_vars_x)]
        
        indices_range_x = np.arange(max_lag_x)
        # print ("##### index")
        # print (indices_range)
        meshgrid_indices_x = np.array(np.meshgrid(*[indices_range_x for _ in range(num_vars_x)], indexing='ij'))
        # print (meshgrid_indices)
        grid_indices_flat_x = meshgrid_indices_x.reshape(num_vars_x, -1).T
        print ("### x ###")

        grid_indices_flat_x_repeat = np.repeat(grid_indices_flat_x, repeats=(max_lag_y-1), axis=0)
        print (pd.DataFrame(grid_indices_flat_x_repeat))
        
        grid_indices_flat_y =  np.array(np.tile(np.arange(1, max_lag_y), (max_lag_x)**num_vars_x)).reshape(-1, 1)
        grid_indices_flat = np.hstack([grid_indices_flat_y, grid_indices_flat_x_repeat])
        grid_np = elements[grid_indices_flat]
        grid_np_2d = grid_np.reshape(grid_np.shape[0], -1)  
        pprint (pd.DataFrame(grid_np_2d))
        
        print ("### y ###")
        print (pd.DataFrame(grid_indices_flat_y))
        
        
        # grid_indices_flat = np.hstack([
        #     grid_indices_flat_y,
        #     np.repeat(grid_indices_flat_x, repeats=(max_lag_y-1), axis=0)
        #     ])       
        # print ("### flat ###")
        # print (pd.DataFrame(grid_indices_flat))
        
        combinations = elements[grid_indices_flat].reshape(-1, (max_lag_x+1)*num_vars_x+max_lag_y)

    
    return (combinations)
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
    model = sm.OLS(y, x).fit()

    y_pred = model.predict(x)
    ehat = y - y_pred
    SSE = np.sum(ehat ** 2)

    return (SSE)

def alt_ols_model (y,x):
    w = np.linalg.inv(x.T @ x) @ x.T @ y

    # Step 2: Predict y using the model
    y_pred = x @ w
    
    # Step 3: Calculate SSE
    sse = np.sum((y - y_pred) ** 2)
    
    return (sse)


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
    "{}_lag{}".format(h, lag) 
    for h in headers_list.tolist()[1:]
    for lag in range(0, max_lags_x + 1) 
    ]
    
    # Lagging x matrix
    n, m = data_matrix_x.shape
    
    lagged_list_x = []
    
    for i in range(m):
        data_var_x = data_matrix_x[:, i].reshape(-1, 1) 
        lagged_var_x = [
            np.vstack((
                np.full((lag, 1), np.nan),
                data_var_x[:-lag]
            )) if lag > 0 else data_var_x  # lag = 0 = matriz original
            for lag in range(0, max_lags_x + 1)
        ]
        lagged_list_x.append (np.hstack(lagged_var_x))
    

    
    lagged_matrix_x = np.hstack(lagged_list_x)


    lagged_matrix_y = np.hstack(lagged_list_y)
  
    lagged_matrix = np.hstack((lagged_matrix_y, lagged_matrix_x))
    
    lagged_headers = lagged_headers_y + lagged_headers_x 

    # print ("### Saving lagged matrix""")
    # pd.DataFrame(lagged_matrix).to_csv ("lagged_matrix.csv", header = lagged_headers)
    
    # print ("### Lagged Matrix ###")
    # print (pd.DataFrame(lagged_matrix, columns = lagged_headers))
       
    # Combinations of posible optimal lags        
    combinations = lag_grid3(data.shape[1], max_lags_y, max_lags_x)
    
    print ("### Lagged Matrix X ###")
    print (pd.DataFrame(lagged_matrix_x))
    
    print ("### Combination ###")
    print (pd.DataFrame(combinations))
    
    
    # for position, combination in enumerate (combinations):
    #     results = lagged_matrix_x[:,combination]
    #     if position == 200:
                
    #         # identity_matrix = np.zeros((n,n))
            
    #         # np.fill_diagonal (identity_matrix, combination)
            
    #         # C = lagged_matrix @ identity_matrix
    #         # print (results)
    #         print (pd.DataFrame(lagged_headers_x).iloc[combination,])
    #         print (pd.DataFrame(results))
    #     else:
    #         continue
        ##########
        
        
    
#     return

# #    q = lags_index_mt (combinations, max_lags_x,min(32, (os.cpu_count() or 1) + 4))
    
#     for i in range ((max_lags_x+1)**6*(max_lags_y)):
#         lagged_matrix = np.hstack((lagged_matrix, lagged_matrix_x))
#     return lagged_matrix
#     # lags_index_y = [
#     # [lag_y# + "-" + str(combination)
#     #  for lag_y in range(combination[0] + 1)]
#     # for combination in combinations
#     # ]
    
#     # # create index    
#     # lags_index = lags_index_y
#     # [lags_index[position].extend(lags_index_x ) for position, lags_index_x in enumerate(lags_index_x)]
    
    
    
    
#     for position, index in enumerate(lags_index):
#         reg_matrix = lagged_matrix[:,index]
#         reg_matrix = reg_matrix[~np.isnan(reg_matrix).any(axis=1)]
        
#         # t11 = time.time()
        
#         # # y = reg_matrix[:,0]
#         # # x = reg_matrix[:,1:]
        
        
        
        
#         # sse_results = alt_ols_model(y, x)
        
        
        
#         # t12 = time.time()
#         # t1n += t12-t11
#     # print (t1n)





# ### Data Marco ###
import os
directory_path = os.path.abspath('')
url = "https://github.com/marco-amh/codes/raw/refs/heads/master/Data.xlsx"
df = pd.read_excel(url, sheet_name = 'Demanda_dinero', index_col = 0)

# df.to_csv ("database_Demanda_dinero.csv")

t1 = time.time()

data_2 = df.copy()
pprint (data_2)

max_lags_y, max_lags_x = 4, 4
lagged_matrix = optimal_lag_selection(data_2, max_lags_y, max_lags_x)

t2 = time.time()

print ("Tiempo de t2-t1: " + str(t2-t1))
