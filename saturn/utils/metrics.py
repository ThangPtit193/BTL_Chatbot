from typing import List

def recall(preds: List[str], targets: List[str]) -> float:
    """
    Calculate the recall score between predicted and target lists.

    Args:
        preds (List[str]): List of predicted values.
        targets (List[str]): List of target values.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """

    if not targets:  # If targets list is empty
        if not preds:  # If preds list is also empty
            return 1.0  # Perfect recall since there are no targets to predict
        else:
            return 0.0  # No recall since there are no targets to predict
        
    true_positive = len(set(preds) & set(targets))
    recall_score = round(true_positive / len(targets), 4)
    
    return recall_score