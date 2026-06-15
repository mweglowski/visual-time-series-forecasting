import torch

def extract_1d_prices(y_image_matrix):
    '''
    Converts 2D probability grid [B, H, W] to 1D price index tracking [B, W]
    Usually used when we have image (matrix) and want to transform into casual form (1D numeric series)
    '''
    return torch.argmax(y_image_matrix, dim=1).float()