"""
CNN Models for Steganalysis
Implements various CNN architectures for detecting steganography in images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC


class StegoEdgeFeatureExtractor(nn.Module):
    """
    Edge detection and feature extraction module for steganalysis.
    Applies multiple edge detection kernels to enhance stego artifacts.
    """
    def __init__(self):
        super().__init__()

        # High-pass filters for edge detection (SRM-like kernels)
        # These help detect subtle changes introduced by steganography
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('laplacian', torch.tensor([
            [[ 0, -1,  0],
             [-1,  4, -1],
             [ 0, -1,  0]]
        ], dtype=torch.float32).unsqueeze(0))

        # High-pass filter for noise residual
        self.register_buffer('highpass', torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract edge features from input images.
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Edge features concatenated with original image [B, C+4, H, W]
        """
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Apply edge detection filters
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)
        laplacian_edges = F.conv2d(gray, self.laplacian, padding=1)
        highpass_edges = F.conv2d(gray, self.highpass, padding=1)

        # Concatenate original image with edge features
        enhanced = torch.cat([x, edges_x, edges_y, laplacian_edges, highpass_edges], dim=1)
        return enhanced


class StegoNet(nn.Module):
    """
    Custom CNN architecture for steganalysis.
    Features:
    - Edge feature extraction
    - Residual connections
    - Batch normalization for stable training
    - Dropout for regularization
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        use_edge_features: bool = True
    ):
        super().__init__()

        self.use_edge_features = use_edge_features

        # Edge feature extractor
        if use_edge_features:
            self.edge_extractor = StegoEdgeFeatureExtractor()
            in_channels = in_channels + 4  # Add 4 edge feature channels

        # First convolutional block - extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second block - deeper features
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third block - high-level features
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fourth block - abstract features
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Extract edge features if enabled
        if self.use_edge_features:
            x = self.edge_extractor(x)

        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Classification
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out


class StegoResNet(nn.Module):
    """
    ResNet-style architecture for steganalysis.
    Uses residual connections for deeper networks.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        use_edge_features: bool = True
    ):
        super().__init__()

        self.use_edge_features = use_edge_features

        if use_edge_features:
            self.edge_extractor = StegoEdgeFeatureExtractor()
            in_channels = in_channels + 4

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_edge_features:
            x = self.edge_extractor(x)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


class StegoLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for steganalysis.
    Handles training, validation, testing, and optimization.
    """
    def __init__(
        self,
        model_type: str = 'stegonet',
        in_channels: int = 3,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.5,
        use_edge_features: bool = True,
        optimizer: str = 'adam',
        scheduler: str = 'cosine',
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        if model_type == 'stegonet':
            self.model = StegoNet(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_edge_features=use_edge_features
            )
        elif model_type == 'stegoresnet':
            self.model = StegoResNet(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                use_edge_features=use_edge_features
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_auroc = AUROC(task='multiclass', num_classes=num_classes)

        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_auroc = AUROC(task='multiclass', num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.val_acc(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(probs, y)

        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_epoch=True)
        self.log('val/recall', self.val_recall, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True)
        self.log('val/auroc', self.val_auroc, on_epoch=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.test_acc(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_f1(preds, y)
        self.test_auroc(probs, y)

        # Log metrics
        self.log('test/loss', loss, on_epoch=True)
        self.log('test/acc', self.test_acc, on_epoch=True)
        self.log('test/precision', self.test_precision, on_epoch=True)
        self.log('test/recall', self.test_recall, on_epoch=True)
        self.log('test/f1', self.test_f1, on_epoch=True)
        self.log('test/auroc', self.test_auroc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Learning rate scheduler
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6
            )
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.hparams.scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


def create_model(config: Optional[Dict] = None) -> StegoLightningModule:
    """
    Factory function to create a steganalysis model.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized Lightning module
    """
    if config is None:
        config = {}

    return StegoLightningModule(
        model_type=config.get('model_type', 'stegonet'),
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 2),
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        dropout_rate=config.get('dropout_rate', 0.5),
        use_edge_features=config.get('use_edge_features', True),
        optimizer=config.get('optimizer', 'adam'),
        scheduler=config.get('scheduler', 'cosine'),
        max_epochs=config.get('max_epochs', 100)
    )
