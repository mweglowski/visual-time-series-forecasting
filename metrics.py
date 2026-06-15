import torch.nn.functional as F
import torch

def jsd_loss(y_pred, y_true):
    '''Sum of column-wise distances between the two images'''
    # Remove dim: [N, 1, H, W] -> [N, H, W]
    y_pred = y_pred.squeeze(1) # channel dim
    y_true = y_true.squeeze(1)
    epsilon = 1e-8 # prevents from division by 0

    # Computing vertical sums
    col_sums_pred = y_pred.sum(dim=1, keepdim=True)
    col_sums_true = y_true.sum(dim=1, keepdim=True)

    # Using these sums to transform cols into discrete probability distributions P and Q
    Q = y_pred / (col_sums_pred + epsilon)
    P = y_true / (col_sums_true + epsilon)
    # Now every column in image is a distribution

    # Get mean distribution
    M = 0.5 * (P + Q)

    # Computing Kullback-Leibler Divergence
    # (M + epsilon) to prevent having log0 which is -inf
    kl_div_p = F.kl_div((M + epsilon).log(), P, reduction='none')
    kl_div_q = F.kl_div((M + epsilon).log(), Q, reduction='none')

    # JSD
    # Sum divergences
    # Get error for each column
    # Get mean for each batch across columns
    # Get mean across batches
    jsd = 0.5 * (kl_div_p + kl_div_q).sum(dim=1).mean(dim=1).mean()
    return jsd

def iou(y_pred, y_true, device):
    '''Compute col-wise 1D IoU'''
    y_pred = y_pred.squeeze(1) # NCHW -> NHW
    y_true = y_true.squeeze(1)
    threshold = 0.1
    batch_size, height, width = y_pred.shape

    # Create height indices [0, 1, ..., height] and reshape to (1, height, 1) - column vector
    row_coords = torch.arange(height, device=device).view(1, height, 1)

    # Generate masks
    pred_mask = y_pred > threshold
    true_mask = y_true > threshold

    # Get max and min coords for future bbox calculation
    pred_coords_max = torch.where(pred_mask, row_coords, 0)
    true_coords_max = torch.where(true_mask, row_coords, 0)
    pred_coords_min = torch.where(pred_mask, row_coords, 999)
    true_coords_min = torch.where(true_mask, row_coords, 999)

    # Extract bbox edges, top and bottom (min, max)
    min_p = pred_coords_min.min(dim=1).values
    max_p = pred_coords_max.max(dim=1).values
    min_t = true_coords_min.min(dim=1).values
    max_t = true_coords_max.max(dim=1).values

    # Intersection
    inter_min = torch.max(min_p, min_t)
    inter_max = torch.min(max_p, max_t)
    # + 1 ensures that if min=3 and max=3, the length is calculated as 1 pixel wide
    intersection = torch.clamp(inter_max - inter_min + 1, min=0) # prevents from intersection lower than 0

    # Union
    union_min = torch.min(min_p, min_t)
    union_max = torch.max(max_p, max_t)
    union = union_max - union_min + 1

    # IoU for all cols
    col_iou = intersection / union

    # Mean across columns, then mean across the whole batch
    return col_iou.mean(dim=1).mean()

def mase(y_pred, y_true, window_len=60):
    y_pred_forecast = y_pred[:, window_len:]
    y_true_forecast = y_true[:, window_len:]
    
    mae_forecast = torch.mean(torch.abs(y_true_forecast - y_pred_forecast))

    y_true_history = y_true[:, :window_len]
    
    yesterday = y_true_history[:, :-1]
    today = y_true_history[:, 1:]
    
    mae_baseline = torch.mean(torch.abs(today - yesterday))

    epsilon = 1e-8
    mase_result = mae_forecast / (mae_baseline + epsilon)
    return mase_result