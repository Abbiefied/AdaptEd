import numpy as np
from sklearn.metrics import average_precision_score

def precision_at_k(y_true, y_pred, k):
    if not y_true or not y_pred:
        return 0.0
    y_pred = y_pred[:k]
    return len(set(y_true) & set(y_pred)) / len(y_pred)

def recall_at_k(y_true, y_pred, k):
    if not y_true or not y_pred:
        return 0.0
    y_pred = y_pred[:k]
    return len(set(y_true) & set(y_pred)) / len(y_true)

def average_precision_score(y_true, y_pred):
    if not y_true or not y_pred:
        return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(y_true)

def mean_average_precision(y_true, y_pred):
    """
    Mean Average Precision
    y_true: list of lists with true relevant items
    y_pred: list of lists with predicted items
    """
    if not y_true or not y_pred:
        return 0.0
    return np.mean([average_precision_score(yt, yp) for yt, yp in zip(y_true, y_pred) if yt and yp])

def evaluate_recommendations(y_true, y_pred, k=10):
    """
    Evaluate recommendations using multiple metrics
    y_true: list of lists with true relevant items
    y_pred: list of lists with predicted items
    k: number of items to consider for Precision@k and Recall@k
    """
    precisions = []
    recalls = []
    average_precisions = []

    for yt, yp in zip(y_true, y_pred):
        if not yt or not yp:
            continue
        
        p = precision_at_k(yt, yp, k)
        r = recall_at_k(yt, yp, k)
        ap = average_precision_score(yt, yp)
        
        precisions.append(p)
        recalls.append(r)
        average_precisions.append(ap)

    if not precisions:
        return {f'precision@{k}': 0, f'recall@{k}': 0, 'MAP': 0}

    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        'MAP': np.mean(average_precisions)
    }