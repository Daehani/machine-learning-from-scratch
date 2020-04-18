import numpy as np

def eval_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred == y_true).mean()