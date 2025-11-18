"""
Quick test script to verify CNN model implementation.
Tests model forward pass, edge features, and basic training loop.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn_models import (
    StegoNet,
    StegoResNet,
    StegoLightningModule,
    StegoEdgeFeatureExtractor
)


def test_edge_feature_extractor():
    """Test edge feature extraction."""
    print("Testing Edge Feature Extractor...")

    extractor = StegoEdgeFeatureExtractor()

    # Create dummy image batch
    batch_size = 4
    channels = 3
    height, width = 256, 256
    x = torch.randn(batch_size, channels, height, width)

    # Extract features
    features = extractor(x)

    # Check output shape (should add 4 edge channels)
    expected_channels = channels + 4
    assert features.shape == (batch_size, expected_channels, height, width), \
        f"Expected shape {(batch_size, expected_channels, height, width)}, got {features.shape}"

    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {features.shape}")
    print(f"  ✓ Edge features extracted successfully!")
    print()


def test_stegonet():
    """Test StegoNet architecture."""
    print("Testing StegoNet Architecture...")

    model = StegoNet(
        in_channels=3,
        num_classes=2,
        dropout_rate=0.5,
        use_edge_features=True
    )

    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, 2), \
        f"Expected shape {(batch_size, 2)}, got {logits.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print(f"  ✓ Output shape: {logits.shape}")
    print()


def test_stegoresnet():
    """Test StegoResNet architecture."""
    print("Testing StegoResNet Architecture...")

    model = StegoResNet(
        in_channels=3,
        num_classes=2,
        dropout_rate=0.5,
        use_edge_features=True
    )

    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, 2), \
        f"Expected shape {(batch_size, 2)}, got {logits.shape}"

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print(f"  ✓ Output shape: {logits.shape}")
    print()


def test_lightning_module():
    """Test PyTorch Lightning module."""
    print("Testing PyTorch Lightning Module...")

    model = StegoLightningModule(
        model_type='stegonet',
        learning_rate=1e-3,
        use_edge_features=True
    )

    # Create dummy batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    y = torch.randint(0, 2, (batch_size,))

    # Test forward pass
    logits = model(x)
    assert logits.shape == (batch_size, 2)

    # Test training step
    batch = (x, y)
    loss = model.training_step(batch, 0)
    assert loss is not None and loss.ndim == 0, "Loss should be a scalar"

    # Test validation step
    model.validation_step(batch, 0)

    # Test optimizer configuration
    optimizer_config = model.configure_optimizers()
    assert 'optimizer' in optimizer_config

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Training step successful")
    print(f"  ✓ Validation step successful")
    print(f"  ✓ Optimizer configured")
    print()


def test_backward_pass():
    """Test backward pass and gradient computation."""
    print("Testing Backward Pass...")

    model = StegoNet(in_channels=3, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Create dummy batch
    x = torch.randn(4, 3, 256, 256)
    y = torch.randint(0, 2, (4,))

    # Forward pass
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients computed!"

    # Optimizer step
    optimizer.step()

    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"  ✓ Gradients computed successfully")
    print(f"  ✓ Optimizer step successful")
    print()


def test_different_image_sizes():
    """Test model with different image sizes."""
    print("Testing Different Image Sizes...")

    model = StegoNet(in_channels=3, num_classes=2)
    model.eval()

    sizes = [128, 256, 512]
    batch_size = 2

    with torch.no_grad():
        for size in sizes:
            x = torch.randn(batch_size, 3, size, size)
            logits = model(x)
            assert logits.shape == (batch_size, 2), \
                f"Failed for size {size}: expected {(batch_size, 2)}, got {logits.shape}"
            print(f"  ✓ Size {size}x{size}: {logits.shape}")

    print()


def test_batch_sizes():
    """Test model with different batch sizes."""
    print("Testing Different Batch Sizes...")

    model = StegoResNet(in_channels=3, num_classes=2)
    model.eval()

    batch_sizes = [1, 4, 16, 32]
    img_size = 256

    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, 3, img_size, img_size)
            logits = model(x)
            assert logits.shape == (bs, 2), \
                f"Failed for batch size {bs}: expected {(bs, 2)}, got {logits.shape}"
            print(f"  ✓ Batch size {bs}: {logits.shape}")

    print()


def test_edge_features_toggle():
    """Test model with and without edge features."""
    print("Testing Edge Features Toggle...")

    # With edge features
    model_with_edges = StegoNet(use_edge_features=True)
    x = torch.randn(2, 3, 256, 256)
    out1 = model_with_edges(x)

    # Without edge features
    model_no_edges = StegoNet(use_edge_features=False)
    out2 = model_no_edges(x)

    assert out1.shape == out2.shape == (2, 2)
    print(f"  ✓ With edge features: {out1.shape}")
    print(f"  ✓ Without edge features: {out2.shape}")
    print()


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running CNN Model Tests")
    print("="*60)
    print()

    try:
        test_edge_feature_extractor()
        test_stegonet()
        test_stegoresnet()
        test_lightning_module()
        test_backward_pass()
        test_different_image_sizes()
        test_batch_sizes()
        test_edge_features_toggle()

        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        print()
        print("Next steps:")
        print("  1. Run training: python scripts/train_cnn.py")
        print("  2. View logs: tensorboard --logdir outputs/logs")
        print("  3. Check outputs in: outputs/checkpoints/")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
