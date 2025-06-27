import numpy as np
import pandas as pd
import statsmodels.api as sm
import time 
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed


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

def fst_ols_model (y,x):
    # Add intercept column (ones)
    w, residuals, rank, s = np.linalg.lstsq(x, y, rcond=None)

    # Predict
    y_pred =x @ w

    # SSE: sum of squared errors
    sse = np.sum((y - y_pred) ** 2)

    return (sse)


def lags_index_fst (combinations, max_lags_x):
    combinations = np.array(combinations)
    num_combos = combinations.shape[0]
    results = []

    for combo_i in range(num_combos):
        combo = combinations[combo_i]
        indices = []

        for pos_x, lags_x in enumerate(combo[1:]):
            base = (max_lags_x + 1) * pos_x
            lag_x_vals = np.arange(lags_x + 1)
            full_index = base + lag_x_vals
            formatted = [f"{idx}-{pos_x}-{tuple(combo)}" for idx in full_index]
            indices.extend(formatted)

        results.append(indices)

    return results

def compute_lags_index_x(combinations, max_lags_x):
    lags_index_x = []
    for comb in combinations:
        pos = np.arange(len(comb) - 1)
        lags = comb[1:]
        for p, l in zip(pos, lags):
            lags_index_x.extend(max_lags_x * p + np.arange(l + 1))
    return lags_index_x

##############
def lags_index (combinations, max_lag_x):
    lags_index_x = [
    [str((max_lags_x+1) * position_x + lag_x) + "-" + str(position_x) + "-" + str(combination)
      for position_x, lags_x in enumerate(combination[1:])
      for lag_x in range(lags_x + 1)]
    for combination in combinations
    ]  
    
    return lags_index_x
###############
def lags_index_2(combinations, max_lags_x):
    # Turn into ndarray for vectorized ops
    combos = np.asarray(combinations, dtype=object)
    results = []

    for combo in combos:
        # Precompute the constant suffix for this combination
        combo_str = str(tuple(combo))
        # Vector of positions (skip the first element of combo)
        positions = np.arange(combo.shape[0] - 1, dtype=int)
        # The corresponding max lag for each position
        lags_per_pos = np.asarray(combo[1:], dtype=int)

        # For each position p, generate an array of strings "idx-p-(combo,...)"
        parts = []
        for p, max_lag in zip(positions, lags_per_pos):
            base = (max_lags_x + 1) * p
            # arange of numeric indices, then cast to string
            idx_str = np.arange(base, base + max_lag + 1).astype(str)
            # build suffix "-p-(combo,...)"
            suffix = f"-{p}-{combo_str}"
            # vectorized concat
            parts.append(np.char.add(idx_str, suffix))

        # concatenate all parts for this combo into one 1-D array
        all_idx = np.concatenate(parts)
        # convert back to Python list of strings
        results.append(all_idx.tolist())

    return results
############################
def _lags_index_single(combo, max_lags_x):
    # Build the "tuple-string" suffix once
    combo_str = str(tuple(combo))
    # Positions correspond to combo[1:], so 0..len(combo)-2
    positions = np.arange(combo.shape[0] - 1, dtype=int)
    lags_per_pos = np.asarray(combo[1:], dtype=int)

    parts = []
    for p, max_lag in zip(positions, lags_per_pos):
        base = (max_lags_x + 1) * p
        # Numeric indices → strings
        idx_str = np.arange(base, base + max_lag + 1).astype(str)
        suffix = f"-{p}-{combo_str}"
        # Vectorized concat in C
        parts.append(np.char.add(idx_str, suffix))

    # Flatten and return as Python list
    return np.concatenate(parts).tolist()

def lags_index_mt(combinations, max_lags_x, max_workers=None):
    """
    Compute lag indices for each combination in parallel threads.
    
    Args:
      combinations : sequence of sequences (e.g. list of tuples/arrays)
      max_lags_x    : int, same as before
      max_workers   : int or None. Threads to use (defaults to os.cpu_count()).
    
    Returns:
      List of lists, each inner list is the lag‐index strings for one combination.
    """
    # Ensure numpy array of objects so each combo is an array
    combos = np.asarray(combinations, dtype=object)

    results = [None] * combos.shape[0]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Schedule one future per combo
        futures = {
            executor.submit(_lags_index_single, combos[i], max_lags_x): i
            for i in range(combos.shape[0])
        }
        # As they complete, store back in results
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results
#####################
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
    
    print (pd.DataFrame(lagged_matrix, columns = lagged_headers))
       
    # Combinations of posible optimal lags        
    combinations = lag_grid(data.shape[1]-1,max_lags_y,max_lags_x)
    
    
#    q = lags_index_mt (combinations, max_lags_x,min(32, (os.cpu_count() or 1) + 4))
    
    for i in range ((max_lags_x+1)**6*(max_lags_y)):
        lagged_matrix = np.hstack((lagged_matrix, lagged_matrix_x))
    return lagged_matrix
    # lags_index_y = [
    # [lag_y# + "-" + str(combination)
    #  for lag_y in range(combination[0] + 1)]
    # for combination in combinations
    # ]
    
    # # create index    
    # lags_index = lags_index_y
    # [lags_index[position].extend(lags_index_x ) for position, lags_index_x in enumerate(lags_index_x)]
    
    
    
    
    for position, index in enumerate(lags_index):
        reg_matrix = lagged_matrix[:,index]
        reg_matrix = reg_matrix[~np.isnan(reg_matrix).any(axis=1)]
        
        # t11 = time.time()
        
        # # y = reg_matrix[:,0]
        # # x = reg_matrix[:,1:]
        
        
        
        
        # sse_results = alt_ols_model(y, x)
        
        
        
        # t12 = time.time()
        # t1n += t12-t11
    # print (t1n)
        
# ### Data Marco ###
import os
directory_path = os.path.abspath('')
url = "https://github.com/marco-amh/codes/raw/refs/heads/master/Data.xlsx"
df = pd.read_excel(url, sheet_name = 'Demanda_dinero', index_col = 0)

t1 = time.time()

data_2 = df.copy()
max_lags_y, max_lags_x = 4, 4
lagged_matrix = optimal_lag_selection(data_2, max_lags_y, max_lags_x)

t2 = time.time()

print ("Tiempo de t2-t1: " + str(t2-t1))

print (lagged_matrix.shape[0])
