import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    """
    Computes the Focal Loss between logits and targets.
    Safe for Automatic Mixed Precision (AMP).
    """
    # Ensure targets have the same data type as logits (e.g., float16 or float32)
    # This prevents "Float can't be cast to Long" errors during autocast
    targets = targets.to(logits.dtype)
    
    # Use binary_cross_entropy_with_logits for numerical stability under AMP.
    # It combines Sigmoid and BCE into a single step.
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Recover the probability p_t using the relationship: bce_loss = -log(p_t)
    # Therefore: p_t = exp(-bce_loss)
    p_t = torch.exp(-bce_loss)
    
    # Apply the focal scaling factor to emphasize hard examples
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
""" def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    \"""
    Original code: https://github.com/apple/ml-destseg/blob/main/model/losses.py#L13
    \"""
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
 """