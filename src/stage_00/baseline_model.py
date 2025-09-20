"""
AWARE-NET Stage 0: EfficientNetV2-S Baseline Model
Balanced baseline implementation with comprehensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import timm
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall

class EfficientNetV2B3Baseline(nn.Module):
    """
    EfficientNetV2 based baseline model for deepfake detection
    
    Features:
    - Pre-trained EfficientNetV2 backbone (supports B0 and rw-t)
    - Binary classification head
    - Dropout for regularization
    - Temperature scaling support for calibration
    
    Supported models:
    - tf_efficientnetv2_b0: 1280 features, 5.9M parameters
    - efficientnetv2_rw_t: 1024 features, 12.6M parameters
    """
    
    def __init__(self, 
                 num_classes: int = 1,  # Changed to 1 for true BCE
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 freeze_backbone: bool = False,
                 model_name: str = 'tf_efficientnetv2_b0'):
        """
        Initialize baseline model for true BCE implementation
        
        Args:
            num_classes: Number of output neurons (1 for true binary classification)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone weights
            model_name: EfficientNetV2 model variant ('tf_efficientnetv2_b0' or 'efficientnetv2_rw_t')
        """
        super().__init__()
        
        # Validate model name
        supported_models = ['tf_efficientnetv2_b0', 'efficientnetv2_rw_t']
        if model_name not in supported_models:
            raise ValueError(f"model_name must be one of {supported_models}, got {model_name}")
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_name = model_name
        
        # Load EfficientNetV2 backbone
        self.backbone = timm.create_model(
            model_name,  # Using selected EfficientNetV2 variant
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=dropout_rate
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Temperature scaling parameter for calibration
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for true BCE implementation
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            Raw logits tensor [batch_size, 1] for BCEWithLogitsLoss
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification (single output for BCE)
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities using sigmoid activation
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities tensor [batch_size, 1] in range [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def forward_with_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temperature scaling
        
        Args:
            x: Input tensor
            
        Returns:
            Temperature-scaled logits
        """
        logits = self.forward(x)
        return logits / self.temperature
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'EfficientNetV2 Baseline ({self.model_name})',
            'backbone': self.model_name,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature.item()
        }

class BaselineTrainer:
    """
    Trainer class for baseline model with comprehensive evaluation
    """
    
    def __init__(self, 
                 model: EfficientNetV2B3Baseline,
                 device: torch.device = torch.device('cuda')):
        """
        Initialize trainer
        
        Args:
            model: Baseline model
            device: Training device
        """
        self.model = model.to(device)
        self.device = device
        
        # Metrics
        self.train_metrics = self._create_metrics()
        self.val_metrics = self._create_metrics()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_auc': [],
            'train_f1': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': [],
            'val_f1': []
        }
    
    def _create_metrics(self) -> Dict[str, any]:
        """Create metric calculators"""
        return {
            'accuracy': Accuracy(task='binary').to(self.device),
            'auc': AUROC(task='binary').to(self.device),
            'f1': F1Score(task='binary').to(self.device),
            'precision': Precision(task='binary').to(self.device),
            'recall': Recall(task='binary').to(self.device)
        }
    
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Reset metrics
        for metric in self.train_metrics.values():
            metric.reset()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(data)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Convert logits to probabilities and predictions for binary classification
            probs = torch.sigmoid(logits.squeeze(1))  # [batch_size]
            preds = (probs > 0.5).long()  # [batch_size]
            
            # Update metric calculators
            for metric in self.train_metrics.values():
                if isinstance(metric, AUROC):
                    metric.update(probs, targets)  # Use sigmoid probabilities
                else:
                    metric.update(preds, targets)
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
        
        # Compute final metrics
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': self.train_metrics['accuracy'].compute().item(),
            'auc': self.train_metrics['auc'].compute().item(),
            'f1': self.train_metrics['f1'].compute().item(),
            'precision': self.train_metrics['precision'].compute().item(),
            'recall': self.train_metrics['recall'].compute().item()
        }
        
        # Update history
        self.history['train_loss'].append(metrics['loss'])
        self.history['train_accuracy'].append(metrics['accuracy'])
        self.history['train_auc'].append(metrics['auc'])
        self.history['train_f1'].append(metrics['f1'])
        
        return metrics
    
    def validate_epoch(self, 
                      dataloader: torch.utils.data.DataLoader,
                      criterion: nn.Module,
                      epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            dataloader: Validation dataloader
            criterion: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        # Reset metrics
        for metric in self.val_metrics.values():
            metric.reset()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits = self.model(data)
                loss = criterion(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                
                # Convert logits to probabilities and predictions for binary classification
                probs = torch.sigmoid(logits.squeeze(1))  # [batch_size]
                preds = (probs > 0.5).long()  # [batch_size]
                
                # Update metric calculators
                for metric in self.val_metrics.values():
                    if isinstance(metric, AUROC):
                        metric.update(probs, targets)  # Use sigmoid probabilities
                    else:
                        metric.update(preds, targets)
        
        # Compute final metrics
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': self.val_metrics['accuracy'].compute().item(),
            'auc': self.val_metrics['auc'].compute().item(),
            'f1': self.val_metrics['f1'].compute().item(),
            'precision': self.val_metrics['precision'].compute().item(),
            'recall': self.val_metrics['recall'].compute().item()
        }
        
        # Update history
        self.history['val_loss'].append(metrics['loss'])
        self.history['val_accuracy'].append(metrics['accuracy'])
        self.history['val_auc'].append(metrics['auc'])
        self.history['val_f1'].append(metrics['f1'])
        
        return metrics
    
    def predict(self, 
               dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate predictions
        
        Args:
            dataloader: Data loader
            
        Returns:
            Tuple of (predictions, probabilities, targets)
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                logits = self.model(data)
                probs = torch.sigmoid(logits.squeeze(1))  # [batch_size]
                preds = (probs > 0.5).long()  # [batch_size]
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())
        
        return (torch.cat(all_preds),
                torch.cat(all_probs),
                torch.cat(all_targets))
    
    def calibrate_temperature(self, 
                            dataloader: torch.utils.data.DataLoader,
                            max_iter: int = 50) -> float:
        """
        Calibrate model using temperature scaling
        
        Args:
            dataloader: Validation dataloader for calibration
            max_iter: Maximum optimization iterations
            
        Returns:
            Optimal temperature value
        """
        self.model.eval()
        
        # Collect logits and targets
        logits_list = []
        targets_list = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits = self.model(data)
                logits_list.append(logits)
                targets_list.append(targets)
        
        logits = torch.cat(logits_list)
        targets = torch.cat(targets_list)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.model.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_func():
            optimizer.zero_grad()
            loss = criterion(logits / self.model.temperature, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_func)
        
        print(f"Temperature calibration complete. Optimal temperature: {self.model.temperature.item():.4f}")
        
        return self.model.temperature.item()

def create_baseline_model(pretrained: bool = True,
                         dropout_rate: float = 0.2,
                         model_name: str = 'tf_efficientnetv2_b0') -> EfficientNetV2B3Baseline:
    """
    Factory function to create baseline model for true BCE
    
    Args:
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate
        model_name: EfficientNetV2 model variant ('tf_efficientnetv2_b0' or 'efficientnetv2_rw_t')
        
    Returns:
        Baseline model instance for true binary classification
    """
    model = EfficientNetV2B3Baseline(
        num_classes=1,  # Changed to 1 for true BCE
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        model_name=model_name
    )
    
    print(f"[OK] Created {model_name} baseline model")
    print(f"   Model info: {model.get_model_info()}")
    
    return model