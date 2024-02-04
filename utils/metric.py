from sklearn.metrics import mean_absolute_percentage_error


def masked_mape(y_true, y_pred):
    idx = y_true != 0
    return mean_absolute_percentage_error(y_true[idx], y_pred[idx])

def f0_mape(y_true, y_pred, denominator=8.0):
    """F0's computing mape formula"""
    error = abs(y_true - y_pred) / denominator
    return error.mean()