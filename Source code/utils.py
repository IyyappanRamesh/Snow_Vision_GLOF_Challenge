import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HybridLoss(nn.Module):
    """Pairing Binary Cross Entropy and Dice Loss for class imbalance mitigation"""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        probs = torch.sigmoid(pred)
        intersection = (probs * target).sum()
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (probs.sum() + target.sum() + self.eps))
        
        return bce_loss + dice_loss

class GlacialLakeDataset(Dataset):
    """Custom standard loader for pairs of satellite imagery and annotations"""
    def __init__(self, image_dir, mask_dir, img_size=(512, 512), transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, filename) # Assuming same name format
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Read mask if it exists (handles active learning loops cleanly)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(self.img_size, dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Normalization and tensor transformation
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
