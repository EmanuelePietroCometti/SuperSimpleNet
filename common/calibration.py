import torch

def piecewise_min_max_calibration(scores_tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Calibrates the scores using piecewise min-max normalization.
    Values below the threshold are linearly scaled to [0, 0.5).
    Values above the threshold are linearly scaled to [0.5, 1.0].
    This ensures the optimal decision boundary becomes exactly 0.5.
    
    Args:
        scores_tensor (torch.Tensor): The raw anomaly scores.
        threshold (float): The optimal threshold calculated via Youden Index.
        
    Returns:
        torch.Tensor: The calibrated scores normalized between 0 and 1.
    """
    scores = scores_tensor.clone().float()
    min_val = scores.min()
    max_val = scores.max()

    if max_val == min_val:
        return scores

    calibrated_scores = torch.zeros_like(scores)

    # Scale lower half: [min_val, threshold) -> [0.0, 0.5)
    lower_mask = scores < threshold
    if lower_mask.any():
        calibrated_scores[lower_mask] = 0.5 * ((scores[lower_mask] - min_val) / (threshold - min_val + 1e-8))

    # Scale upper half: [threshold, max_val] -> [0.5, 1.0]
    upper_mask = scores >= threshold
    if upper_mask.any():
        calibrated_scores[upper_mask] = 0.5 + 0.5 * ((scores[upper_mask] - threshold) / (max_val - threshold + 1e-8))

    return calibrated_scores.clip(0, 1)