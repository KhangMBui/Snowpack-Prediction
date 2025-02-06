import numpy as np

def nash_sutcliffe_efficiency(observed, predicted):
    """
    Calculate the Nash Sutcliffe Efficiency (NSE).
    
    Parameters:
    observed (array-like): Observed values
    predicted (array-like): Predicted values
    
    Returns:
    float: NSE value
    """
    observed_mean = np.mean(observed)
    numerator = np.sum((predicted - observed) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def relative_bias(observed, predicted):
    """
    Calculate the Relative Bias (%).
    
    Parameters:
    observed (array-like): Observed values
    predicted (array-like): Predicted values
    
    Returns:
    float: Relative Bias value
    """
    numerator = np.sum(predicted - observed)
    denominator = np.sum(observed)
    rel_bias = (numerator / denominator) * 100
    return rel_bias

def actual_error(observed, predicted):
    """
    Calculate the Actual Error (Prediction - Observed).
    
    Parameters:
    observed (array-like): Observed values
    predicted (array-like): Predicted values
    
    Returns:
    array: Actual error for each data point
    """
    return predicted - observed


def root_mean_square_error(observed, predicted):
    """
    Calculate the Root Mean Square Error (RMSE).
    
    Parameters:
    observed (array-like): Observed values
    predicted (array-like): Predicted values
    
    Returns:
    float: RMSE value
    """
    mse = np.mean((predicted - observed) ** 2)
    rmse = np.sqrt(mse)
    return rmse