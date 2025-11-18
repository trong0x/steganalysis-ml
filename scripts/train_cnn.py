"""
Training script for CNN steganalysis models.
Includes dummy data generation for testing and hyperparameter tuning.
"""

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

# Import the model (assuming it's in src/models/cnn_models.py)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.models.cnn_models import StegoLightningModule, create_model


class DummyStegoDataset(Dataset):
    """
    Dummy dataset for testing the steganalysis pipeline.
    Generates synthetic images with realistic noise patterns.
    """
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 256,
        num_channels: int = 3,
        num_classes: int = 2,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

        if seed is not None:
            torch.manual_seed(seed)

        # Generate random images
        self.images = torch.randn(num_samples, num_channels, image_size, image_size)

        # Add patterns to make stego/cover distinguishable for testing
        for i in range(num_samples):
            if i % 2 == 0:  # Cover images (class 0)
                # Add smooth gradients
                x = torch.linspace(0, 1, image_size).view(1, -1)
                y = torch.linspace(0, 1, image_size).view(-1, 1)
                gradient = (x + y) / 2
                self.images[i] += gradient.unsqueeze(0) * 0.5
            else:  # Stego images (class 1)
                # Add subtle high-frequency noise (simulating stego artifacts)
                noise = torch.randn(num_channels, image_size, image_size) * 0.1
                self.images[i] += noise

        # Normalize to [0, 1]
        self.images = (self.images - self.images.min()) / (self.images.max() - self.images.min())

        # Generate labels
        self.labels = torch.tensor([i % num_classes for i in range(num_samples)], dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def create_dataloaders(
    num_train: int = 800,
    num_val: int = 100,
    num_test: int = 100,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    seed: int = 42
):
    """
    Create train, validation, and test dataloaders with dummy data.
    """
    print(f"Creating dummy datasets...")
    print(f"  Train: {num_train} samples")
    print(f"  Val: {num_val} samples")
    print(f"  Test: {num_test} samples")

    # Create datasets
    train_dataset = DummyStegoDataset(
        num_samples=num_train,
        image_size=image_size,
        seed=seed
    )

    val_dataset = DummyStegoDataset(
        num_samples=num_val,
        image_size=image_size,
        seed=seed + 1
    )

    test_dataset = DummyStegoDataset(
        num_samples=num_test,
        image_size=image_size,
        seed=seed + 2
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_model(args):
    """Main training function."""

    # Set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed
    )

    # Create model
    print(f"\nInitializing {args.model_type} model...")
    model_config = {
        'model_type': args.model_type,
        'in_channels': 3,
        'num_classes': 2,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'use_edge_features': args.use_edge_features,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'max_epochs': args.max_epochs
    }

    model = create_model(model_config)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    callbacks = [
        # Model checkpoint - save best model
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        # Early stopping
        EarlyStopping(
            monitor='val/loss',
            patience=args.early_stop_patience,
            mode='min',
            verbose=True
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs',
        version=args.experiment_name
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None
    )

    # Train
    print(f"\nStarting training for {args.max_epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')

    # Print results
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    for key, value in test_results[0].items():
        print(f"{key}: {value:.4f}")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    print(f"\nBest model checkpoint: {checkpoint_dir / 'best_model.ckpt'}")
    print(f"TensorBoard logs: {output_dir / 'logs'}")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {output_dir / 'logs'}")

    return model, test_results


def main():
    parser = argparse.ArgumentParser(description='Train CNN for steganalysis')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='stegonet',
                        choices=['stegonet', 'stegoresnet'],
                        help='Model architecture to use')
    parser.add_argument('--use_edge_features', action='store_true', default=True,
                        help='Use edge detection features')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')

    # Data arguments
    parser.add_argument('--num_train', type=int, default=800,
                        help='Number of training samples')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--num_test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (square)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'reduce_on_plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--gradient_clip_val', type=float, default=0.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Early stopping patience')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='stego_cnn',
                        help='Experiment name for logging')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Print configuration
    print("="*50)
    print("Training Configuration")
    print("="*50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)

    # Train model
    train_model(args)


if __name__ == '__main__':
    main()
