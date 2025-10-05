"""
AWARE-NET Stage 0: CelebDF Dataset Implementation
PyTorch dataset class for CelebDF-v2 with preprocessing
"""

import os
import logging
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


logger = logging.getLogger(__name__)

class CelebDFDataset(Dataset):
    """
    CelebDF-v2 Dataset for deepfake detection

    Features:
    - Loads images from manifest CSV files
    - Supports train/val/test splits
    - Configurable data augmentation
    - Proper image preprocessing
    - Path leakage protection (anonymized paths)
    """

    def __init__(self,
                 manifest_path: str,
                 root_path: str = ".",
                 image_size: int = 256,
                 augmentation: bool = True,
                 normalize: bool = True,
                 path_leakage_check: bool = True):
        """
        Initialize dataset

        Args:
            manifest_path: Path to CSV manifest file
            root_path: Root directory for relative paths
            image_size: Target image size (will be resized to image_size x image_size)
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize images
            path_leakage_check: Whether to check for potential path leakage
        """
        self.manifest_path = Path(manifest_path)
        self.root_path = Path(root_path)
        self.image_size = image_size
        self.augmentation = augmentation
        self.path_leakage_check = path_leakage_check
        self.normalize = normalize
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Load manifest
        self.df = pd.read_csv(self.manifest_path)

        # Check for potential path leakage
        if self.path_leakage_check:
            self._check_path_leakage()
        
        # Filter valid samples
        self.df = self.df[self.df.get('valid', True) == True]
        
        logger.info(
            "Loaded manifest %s: total=%d real=%d fake=%d",
            self.manifest_path,
            len(self.df),
            (self.df['label'] == 0).sum(),
            (self.df['label'] == 1).sum()
        )
        
        # Setup transforms
        self._setup_transforms()

    def _check_path_leakage(self):
        """
        Check for potential path leakage in the dataset

        Warns if paths contain obvious label indicators
        """
        leakage_indicators = ['real', 'fake', 'authentic', 'forged', 'true', 'false']
        potential_leakage = []

        for idx, row in self.df.iterrows():
            path_lower = str(row['image_path']).lower()
            for indicator in leakage_indicators:
                if indicator in path_lower:
                    potential_leakage.append((row['image_path'], indicator, row['label']))
                    break

        if potential_leakage:
            logger.warning(
                "Potential path leakage detected in %d samples. This may inflate metrics.",
                len(potential_leakage)
            )
            for path, indicator, label in potential_leakage[:5]:
                logger.warning("Example leakage: %s (contains '%s', label=%s)", path, indicator, label)
            logger.warning("Consider using anonymized manifests to prevent leakage")
        else:
            logger.info("Path leakage check passed for %s", self.manifest_path)

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
            
            # Convert label to tensor - use float for BCE compatibility
            label_tensor = torch.tensor(label, dtype=torch.float)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error("Error loading image %s: %s", image_path, e)
            # Return a blank image and label
            blank_image = torch.zeros(3, self.image_size, self.image_size)
            return blank_image, torch.tensor(0, dtype=torch.float)
    
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
                       dataset_name: str = 'celebdf_v2',
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
    config = DatasetConfig(config_path, dataset_name=dataset_name)
    
    # Get manifest paths and validate
    train_manifest = config.get_manifest_path('train')
    val_manifest = config.get_manifest_path('val')
    test_manifest = config.get_manifest_path('test')

    for manifest in (train_manifest, val_manifest, test_manifest):
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest file missing: {manifest}")

    logger.info("Loading dataset %s from %s", dataset_name, config_path)
    logger.info("  Train manifest: %s", train_manifest)
    logger.info("  Val manifest: %s", val_manifest)
    logger.info("  Test manifest: %s", test_manifest)
    
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
    
    logger.info("Dataset information:")
    for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
        info = dataset.get_dataset_info()
        logger.info(
            "%s → total=%d real=%d fake=%d balance=%.3f",
            name,
            info['total_samples'],
            info['real_samples'],
            info['fake_samples'],
            info['balance_ratio']
        )
    
    return train_loader, val_loader, test_loader

def test_dataset_loading():
    """Test dataset loading functionality."""
    logger.info("Testing CelebDF dataset loading with manifest configuration")

    config_path = "configs/datasets.json"

    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config_path=config_path,
            dataset_name='celebdf_v2',
            batch_size=4,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )

        logger.info("Testing data loader batches")
        for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
            batch_images, batch_labels = next(iter(loader))
            logger.info(
                "%s batch → shape=%s labels_shape=%s range=[%.3f, %.3f] labels=%s",
                name,
                tuple(batch_images.shape),
                tuple(batch_labels.shape),
                batch_images.min().item(),
                batch_images.max().item(),
                batch_labels.tolist(),
            )

        logger.info("Dataset loading smoke test completed successfully")
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("Dataset loading test failed: %s", exc)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the dataset implementation
    test_dataset_loading()
