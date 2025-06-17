# Autoselection of optimal lag selection 

This repository provides a Python implementation of an algorithm for **automatic lag selection in Autoregressive Distributed Lag (ARDL) models**. The tool evaluates all possible combination of lag selection and returns the optimal lag selection with AIC and BIC. The tool is designed to assist on research in identifying the optimal lag structure for ARDL models, facilitating time series analysis and econometric modeling. 

## Features
- Automatic selection of lags for both endogenous and exogenous variables.
- Support for standard information criteria (AIC, BIC) for lag selection.
- Efficient computation using NumPy and statsmodels.
- Easy integration into existing time series workflows.
  
## Module Versions
The algorithm was developed and tested using the following versions:

- Python: 3.8.8
- NumPy: 1.23.x
- Statsmodels: 0.14.x
- Pandas: 2.0.x

## Input Parameters
- max_lag_y: Maximum lag order allowed for the endogenous variable.
- max_lag_x: Maximum lag order allowed for the exogenous variables.
- data: DataFrame where the first column contains the endogenous variable, and the second column onward contain the exogenous variables.

## Output
- Array with optimal selection using the AIC and BIC.
