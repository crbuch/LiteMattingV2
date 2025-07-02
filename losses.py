import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def alpha_loss(pred_alpha, gt_alpha, trimap):
    """
    Alpha prediction loss (L1 loss in unknown regions)
    """
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()  # Unknown region mask
    
    # L1 loss in unknown regions
    diff = torch.abs(pred_alpha - gt_alpha)
    loss = torch.sum(diff * unknown_mask) / (torch.sum(unknown_mask) + 1e-8)
    
    return loss


def composition_loss(pred_alpha, gt_alpha, image, trimap):
    """
    Composition loss - compares composited images
    """
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    
    # Composite with predicted alpha
    pred_comp = pred_alpha * image
    gt_comp = gt_alpha * image
    
    # L1 loss on composition in unknown regions
    diff = torch.abs(pred_comp - gt_comp)
    loss = torch.sum(diff * unknown_mask) / (torch.sum(unknown_mask) + 1e-8)
    
    return loss


def gradient_loss(pred_alpha, gt_alpha, trimap):
    """
    Gradient loss to preserve fine details
    """
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    
    # Sobel filters for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=pred_alpha.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=pred_alpha.device).view(1, 1, 3, 3)
    
    # Compute gradients
    pred_grad_x = F.conv2d(pred_alpha, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_alpha, sobel_y, padding=1)
    gt_grad_x = F.conv2d(gt_alpha, sobel_x, padding=1)
    gt_grad_y = F.conv2d(gt_alpha, sobel_y, padding=1)
    
    # Gradient magnitude
    pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
    gt_grad = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-8)
    
    # L1 loss on gradients in unknown regions
    diff = torch.abs(pred_grad - gt_grad)
    loss = torch.sum(diff * unknown_mask) / (torch.sum(unknown_mask) + 1e-8)
    
    return loss


def laplacian_loss(pred_alpha, gt_alpha, trimap):
    """
    Laplacian loss for smoothness
    """
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    
    # Laplacian kernel
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                   dtype=torch.float32, device=pred_alpha.device).view(1, 1, 3, 3)
    
    # Apply Laplacian
    pred_lap = F.conv2d(pred_alpha, laplacian_kernel, padding=1)
    gt_lap = F.conv2d(gt_alpha, laplacian_kernel, padding=1)
    
    # L1 loss on Laplacian in unknown regions
    diff = torch.abs(pred_lap - gt_lap)
    loss = torch.sum(diff * unknown_mask) / (torch.sum(unknown_mask) + 1e-8)
    
    return loss


class MattingLoss(nn.Module):
    """
    Combined matting loss function
    """
    def __init__(self, alpha_weight=1.0, comp_weight=1.0, grad_weight=1.0, lap_weight=1.0):
        super(MattingLoss, self).__init__()
        self.alpha_weight = alpha_weight
        self.comp_weight = comp_weight
        self.grad_weight = grad_weight
        self.lap_weight = lap_weight
    
    def forward(self, pred_alpha, gt_alpha, image, trimap):
        """
        Compute combined matting loss
        
        Args:
            pred_alpha: Predicted alpha matte (B, 1, H, W)
            gt_alpha: Ground truth alpha matte (B, 1, H, W)
            image: Input image (B, 3, H, W)
            trimap: Trimap (B, 3, H, W) - [bg, unknown, fg]
        """
        losses = {}
        
        # Alpha prediction loss
        losses['alpha'] = alpha_loss(pred_alpha, gt_alpha, trimap)
        
        # Composition loss
        losses['composition'] = composition_loss(pred_alpha, gt_alpha, image, trimap)
        
        # Gradient loss
        losses['gradient'] = gradient_loss(pred_alpha, gt_alpha, trimap)
        
        # Laplacian loss
        losses['laplacian'] = laplacian_loss(pred_alpha, gt_alpha, trimap)
        
        # Total loss
        total_loss = (self.alpha_weight * losses['alpha'] + 
                     self.comp_weight * losses['composition'] + 
                     self.grad_weight * losses['gradient'] + 
                     self.lap_weight * losses['laplacian'])
        
        losses['total'] = total_loss
        
        return losses


# Evaluation metrics
def compute_sad(pred_alpha, gt_alpha, trimap):
    """Sum of Absolute Differences"""
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    diff = torch.abs(pred_alpha - gt_alpha) * unknown_mask
    sad = torch.sum(diff, dim=[1, 2, 3]) / 1000.0  # Scale to thousands
    return sad.mean()


def compute_mse(pred_alpha, gt_alpha, trimap):
    """Mean Squared Error in unknown regions"""
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    diff = (pred_alpha - gt_alpha) ** 2 * unknown_mask
    mse = torch.sum(diff, dim=[1, 2, 3]) / torch.sum(unknown_mask, dim=[1, 2, 3])
    return mse.mean()


def compute_gradient_error(pred_alpha, gt_alpha, trimap):
    """Gradient error in unknown regions"""
    unknown_mask = (trimap[:, 1:2, :, :] > 0.5).float()
    
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=pred_alpha.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=pred_alpha.device).view(1, 1, 3, 3)
    
    # Compute gradients
    pred_grad_x = F.conv2d(pred_alpha, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred_alpha, sobel_y, padding=1)
    gt_grad_x = F.conv2d(gt_alpha, sobel_x, padding=1)
    gt_grad_y = F.conv2d(gt_alpha, sobel_y, padding=1)
    
    # Gradient error
    grad_error = torch.sqrt((pred_grad_x - gt_grad_x)**2 + (pred_grad_y - gt_grad_y)**2 + 1e-8)
    grad_error = grad_error * unknown_mask
    
    error = torch.sum(grad_error, dim=[1, 2, 3]) / (torch.sum(unknown_mask, dim=[1, 2, 3]) + 1e-8)
    return error.mean() / 1000.0  # Scale to thousands
