"""
AWARE-NET Stage 0: CelebDF Dataset Implementation
PyTorch dataset class for CelebDF-v2 with preprocessing
"""

import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CelebDFDataset(Dataset):
    """
    CelebDF-v2 Dataset for deepfake detection
    
    Features:
    - Loads images from manifest CSV files
    - Supports train/val/test splits
    - Configurable data augmentation
    - Proper image preprocessing
    """
    
    def __init__(self, 
                 manifest_path: str,
                 root_path: str = ".",
                 image_size: int = 256,
                 augmentation: bool = True,
                 normalize: bool = True):
        """
        Initialize dataset
        
        Args:
            manifest_path: Path to CSV manifest file
            root_path: Root directory for relative paths
            image_size: Target image size (will be resized to image_size x image_size)
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
        """
        self.manifest_path = Path(manifest_path)
        self.root_path = Path(root_path)
        self.image_size = image_size
        self.augmentation = augmentation
        self.normalize = normalize
        
        # Load manifest
        self.df = pd.read_csv(self.manifest_path)
        
        # Filter valid samples
        self.df = self.df[self.df.get('valid', True) == True]
        
        print(f"Loaded {len(self.df)} samples from {self.manifest_path}")
        print(f"Real samples: {(self.df['label'] == 0).sum()}")
        print(f"Fake samples: {(self.df['label'] == 1).sum()}")
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image transforms using albumentations"""
        
        # Base transforms (always applied)
        base_transforms = [
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR),
        ]
        
        # Training augmentations
        if self.augmentation:
            aug_transforms = [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.1, 
                    hue=0.05, 
                    p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.2
                ),
            ]
            base_transforms.extend(aug_transforms)
        
        # Normalization
        if self.normalize:
            # ImageNet normalization
            base_transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                )
            )
        else:
            # Just convert to 0-1 range
            base_transforms.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0))
        
        # Convert to tensor
        base_transforms.append(ToTensorV2())
        
        self.transform = A.Compose(base_transforms)
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Get sample info
        row = self.df.iloc[idx]
        image_path = self.root_path / row['image_path']
        label = int(row['label'])
        
        # Load image
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            # Convert label to tensor
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image and label
            blank_image = torch.zeros(3, self.image_size, self.image_size)
            return blank_image, torch.tensor(0, dtype=torch.long)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for balanced training
        
        Returns:
            Tensor with class weights
        """
        real_count = (self.df['label'] == 0).sum()
        fake_count = (self.df['label'] == 1).sum()
        total = real_count + fake_count
        
        # Inverse frequency weighting
        real_weight = total / (2 * real_count)
        fake_weight = total / (2 * fake_count)
        
        return torch.tensor([real_weight, fake_weight], dtype=torch.float)
    
    def get_class_counts(self) -> torch.Tensor:
        """
        Get actual class counts for pos_weight calculation in BCE
        
        Returns:
            Tensor with [real_count, fake_count]
        """
        real_count = (self.df['label'] == 0).sum()
        fake_count = (self.df['label'] == 1).sum()
        
        return torch.tensor([real_count, fake_count], dtype=torch.float)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        real_count = (self.df['label'] == 0).sum()
        fake_count = (self.df['label'] == 1).sum()
        
        return {
            'total_samples': len(self.df),
            'real_samples': real_count,
            'fake_samples': fake_count,
            'balance_ratio': min(real_count, fake_count) / max(real_count, fake_count),
            'image_size': self.image_size,
            'augmentation': self.augmentation,
            'normalize': self.normalize
        }

def create_data_loaders(config_path: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config_path: Path to dataset configuration
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Import here to avoid circular imports
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.dataset_config import DatasetConfig
    
    # Load configuration
    config = DatasetConfig(config_path)
    
    # Get manifest paths
    train_manifest = config.get_manifest_path('train')
    val_manifest = config.get_manifest_path('val')
    test_manifest = config.get_manifest_path('test')
    
    print(f"Loading datasets from:")
    print(f"  Train: {train_manifest}")
    print(f"  Val: {val_manifest}")
    print(f"  Test: {test_manifest}")
    
    # Create datasets
    train_dataset = CelebDFDataset(
        manifest_path=train_manifest,
        root_path=config.root_path,
        image_size=config.metadata.image_size[0],
        augmentation=True,  # Enable augmentation for training
        normalize=True
    )
    
    val_dataset = CelebDFDataset(
        manifest_path=val_manifest,
        root_path=config.root_path,
        image_size=config.metadata.image_size[0],
        augmentation=False,  # No augmentation for validation
        normalize=True
    )
    
    test_dataset = CelebDFDataset(
        manifest_path=test_manifest,
        root_path=config.root_path,
        image_size=config.metadata.image_size[0],
        augmentation=False,  # No augmentation for testing
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Print dataset information
    print("\\nDataset Information:")
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        info = dataset.get_dataset_info()
        print(f"{name}: {info['total_samples']} samples ({info['real_samples']} real, {info['fake_samples']} fake)")
        print(f"  Balance ratio: {info['balance_ratio']:.3f}")
    
    return train_loader, val_loader, test_loader

def test_dataset_loading():
    """Test dataset loading functionality"""
    print("Testing CelebDF Dataset Loading...")
    
    # Test with a small sample
    config_path = "configs/dataset_paths.json"
    
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config_path=config_path,
            batch_size=4,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        print("\\nTesting data loading...")
        
        # Test one batch from each loader
        for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
            batch_images, batch_labels = next(iter(loader))
            print(f"{name} batch:")
            print(f"  Images shape: {batch_images.shape}")
            print(f"  Labels shape: {batch_labels.shape}")
            print(f"  Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
            print(f"  Labels: {batch_labels.tolist()}")
        
        print("\\nDataset loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Dataset loading test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the dataset implementation
    test_dataset_loading()