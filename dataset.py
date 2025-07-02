import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class Composition1KDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None, input_size=512):
        """
        Adobe Composition-1K Dataset
        
        Args:
            data_root: Path to Composition-1K dataset
            mode: 'train' or 'test'
            transform: Data augmentation transforms
            input_size: Input image size for training
        """
        self.data_root = data_root
        self.mode = mode
        self.input_size = input_size
        self.transform = transform
        
        if mode == 'train':
            self.fg_path = os.path.join(data_root, 'Training_set', 'fg')
            self.alpha_path = os.path.join(data_root, 'Training_set', 'alpha')
            self.bg_path = os.path.join(data_root, 'Training_set', 'bg')
            self.composite_path = os.path.join(data_root, 'Training_set', 'merged')
            self.trimap_path = os.path.join(data_root, 'Training_set', 'trimaps')
        else:
            self.fg_path = os.path.join(data_root, 'Test_set', 'fg')
            self.alpha_path = os.path.join(data_root, 'Test_set', 'alpha')
            self.bg_path = os.path.join(data_root, 'Test_set', 'bg')
            self.composite_path = os.path.join(data_root, 'Test_set', 'merged')
            self.trimap_path = os.path.join(data_root, 'Test_set', 'trimaps')
        
        # Get list of images
        self.image_list = []
        if os.path.exists(self.composite_path):
            self.image_list = sorted([f for f in os.listdir(self.composite_path) 
                                    if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Found {len(self.image_list)} images in {mode} set")
    
    def __len__(self):
        return len(self.image_list)
    
    def load_image(self, path):
        """Load image and convert to RGB"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_alpha(self, path):
        """Load alpha matte"""
        alpha = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if alpha is None:
            raise ValueError(f"Could not load alpha: {path}")
        return alpha.astype(np.float32) / 255.0
    
    def generate_trimap(self, alpha, k_size=10):
        """Generate trimap from alpha matte"""
        # Convert alpha to 0-255 range
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        # Create erosion and dilation kernels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # Erode and dilate to create unknown region
        eroded = cv2.erode(alpha_uint8, kernel, iterations=1)
        dilated = cv2.dilate(alpha_uint8, kernel, iterations=1)
        
        # Create trimap: 0=background, 128=unknown, 255=foreground
        trimap = np.zeros_like(alpha_uint8)
        trimap[eroded > 128] = 255  # Foreground
        trimap[(dilated > 128) & (eroded <= 128)] = 128  # Unknown
        # Background remains 0
        
        return trimap
    
    def trimap_to_clicks(self, trimap):
        """Convert trimap to click guidance maps"""
        h, w = trimap.shape
        clicks = np.zeros((h, w, 2), dtype=np.float32)
        
        # Distance transform for foreground and background
        fg_mask = (trimap == 255).astype(np.uint8)
        bg_mask = (trimap == 0).astype(np.uint8)
        
        if np.sum(fg_mask) > 0:
            dt_fg = cv2.distanceTransform(1 - fg_mask, cv2.DIST_L2, 0)
            clicks[:, :, 0] = np.exp(-dt_fg**2 / (2 * (0.05 * 320)**2))
        
        if np.sum(bg_mask) > 0:
            dt_bg = cv2.distanceTransform(1 - bg_mask, cv2.DIST_L2, 0)
            clicks[:, :, 1] = np.exp(-dt_bg**2 / (2 * (0.05 * 320)**2))
        
        return clicks
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        base_name = os.path.splitext(image_name)[0]
        
        # Load composite image
        composite_path = os.path.join(self.composite_path, image_name)
        composite = self.load_image(composite_path)
        
        # Load alpha matte
        alpha_path = os.path.join(self.alpha_path, image_name)
        if not os.path.exists(alpha_path):
            alpha_path = os.path.join(self.alpha_path, base_name + '.png')
        alpha = self.load_alpha(alpha_path)
        
        # Load or generate trimap
        trimap_file = os.path.join(self.trimap_path, base_name + '.png')
        if os.path.exists(trimap_file):
            trimap = cv2.imread(trimap_file, cv2.IMREAD_GRAYSCALE)
        else:
            # Generate trimap from alpha if trimap doesn't exist
            trimap = self.generate_trimap(alpha)
        
        # Convert trimap to one-hot encoding
        trimap_onehot = np.zeros((trimap.shape[0], trimap.shape[1], 3), dtype=np.float32)
        trimap_onehot[:, :, 0] = (trimap == 0).astype(np.float32)    # Background
        trimap_onehot[:, :, 1] = (trimap == 128).astype(np.float32)  # Unknown
        trimap_onehot[:, :, 2] = (trimap == 255).astype(np.float32)  # Foreground
        
        # Generate click guidance
        clicks = self.trimap_to_clicks(trimap)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(
                image=composite,
                mask=alpha,
                masks=[trimap_onehot[:,:,0], trimap_onehot[:,:,1], trimap_onehot[:,:,2], 
                       clicks[:,:,0], clicks[:,:,1]]
            )
            composite = transformed['image']
            alpha = transformed['mask']
            trimap_bg, trimap_unk, trimap_fg, click_fg, click_bg = transformed['masks']
            
            trimap_onehot = np.stack([trimap_bg, trimap_unk, trimap_fg], axis=2)
            clicks = np.stack([click_fg, click_bg], axis=2)
        
        # Convert to tensors
        if not torch.is_tensor(composite):
            composite = torch.from_numpy(composite.transpose(2, 0, 1)).float() / 255.0
        if not torch.is_tensor(alpha):
            alpha = torch.from_numpy(alpha).unsqueeze(0).float()
        
        trimap_tensor = torch.from_numpy(trimap_onehot.transpose(2, 0, 1)).float()
        clicks_tensor = torch.from_numpy(clicks.transpose(2, 0, 1)).float()
        
        return {
            'image': composite,
            'alpha': alpha,
            'trimap': trimap_tensor,
            'clicks': clicks_tensor,
            'name': image_name
        }


def get_train_transforms(input_size=512):
    """Get training data augmentation transforms"""
    return A.Compose([
        A.RandomResizedCrop(input_size, input_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ], additional_targets={
        'mask': 'mask',
        'masks': 'masks'
    })


def get_val_transforms(input_size=512):
    """Get validation transforms"""
    return A.Compose([
        A.Resize(input_size, input_size),
    ], additional_targets={
        'mask': 'mask',
        'masks': 'masks'
    })


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = torch.stack([item['image'] for item in batch])
    alphas = torch.stack([item['alpha'] for item in batch])
    trimaps = torch.stack([item['trimap'] for item in batch])
    clicks = torch.stack([item['clicks'] for item in batch])
    names = [item['name'] for item in batch]
    
    return {
        'image': images,
        'alpha': alphas,
        'trimap': trimaps,
        'clicks': clicks,
        'name': names
    }
