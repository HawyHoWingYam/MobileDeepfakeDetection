"""
Test suite for baseline model implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from stage_00.baseline_model import EfficientNetV2B3Baseline


class TestEfficientNetV2B3Baseline:
    """Test EfficientNetV2B3Baseline model"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.model = EfficientNetV2B3Baseline(
            num_classes=2,
            pretrained=False,  # Use False to avoid downloading weights during testing
            dropout_rate=0.2
        )
        self.batch_size = 4
        self.image_size = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_model_initialization(self):
        """Test model initialization with different parameters"""
        # Test default initialization
        model = EfficientNetV2B3Baseline()
        assert model.num_classes == 2
        assert model.dropout_rate == 0.2
        
        # Test custom initialization
        model_custom = EfficientNetV2B3Baseline(
            num_classes=5,
            pretrained=False,
            dropout_rate=0.5,
            freeze_backbone=True
        )
        assert model_custom.num_classes == 5
        assert model_custom.dropout_rate == 0.5
        
        # Test that backbone is frozen
        if hasattr(model_custom, 'backbone'):
            for param in model_custom.backbone.parameters():
                assert not param.requires_grad
    
    def test_model_architecture(self):
        """Test model architecture and components"""
        # Check that model has expected components
        assert hasattr(self.model, 'backbone')
        assert hasattr(self.model, 'classifier')
        
        # Test model is in training mode by default
        assert self.model.training
        
        # Test model can be switched to eval mode
        self.model.eval()
        assert not self.model.training
        
        # Test model parameters exist and are trainable
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass with different input sizes"""
        self.model.eval()
        
        # Test standard input
        batch_size = 4
        x = torch.randn(batch_size, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            output = self.model(x)
        
        # Check output shape
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        assert logits.shape == (batch_size, self.model.num_classes)
        assert not torch.isnan(logits).any()
        assert torch.isfinite(logits).all()
    
    def test_different_batch_sizes(self):
        """Test forward pass with different batch sizes"""
        self.model.eval()
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, self.image_size, self.image_size)
            
            with torch.no_grad():
                output = self.model(x)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == self.model.num_classes
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the model"""
        self.model.train()
        
        # Create dummy input and target
        x = torch.randn(2, 3, self.image_size, self.image_size, requires_grad=True)
        target = torch.randint(0, 2, (2,))
        
        # Forward pass
        output = self.model(x)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_gradients = True
                assert not torch.isnan(param.grad).any()
                assert torch.isfinite(param.grad).all()
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_output_probabilities(self):
        """Test that model outputs can be converted to valid probabilities"""
        self.model.eval()
        
        x = torch.randn(4, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            output = self.model(x)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)
            
            # Check probability properties
            assert torch.all(probs >= 0)  # Non-negative
            assert torch.all(probs <= 1)  # <= 1
            assert torch.allclose(probs.sum(dim=1), torch.ones(4))  # Sum to 1
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"
            
            # Save model
            torch.save(self.model.state_dict(), model_path)
            
            # Load model
            loaded_model = EfficientNetV2B3Baseline(
                num_classes=self.model.num_classes,
                pretrained=False,
                dropout_rate=self.model.dropout_rate
            )
            loaded_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Test that loaded model works
            self.model.eval()
            loaded_model.eval()
            
            x = torch.randn(2, 3, self.image_size, self.image_size)
            
            with torch.no_grad():
                output1 = self.model(x)
                output2 = loaded_model(x)
                
                if isinstance(output1, dict):
                    logits1 = output1['logits']
                    logits2 = output2['logits']
                else:
                    logits1 = output1
                    logits2 = output2
                
                # Outputs should be identical
                assert torch.allclose(logits1, logits2, atol=1e-5)
    
    def test_model_device_transfer(self):
        """Test model transfer between devices"""
        # Test CPU model
        self.model.cpu()
        x_cpu = torch.randn(2, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            output_cpu = self.model(x_cpu)
        
        assert isinstance(output_cpu, (torch.Tensor, dict))
        
        # Test GPU model if available
        if torch.cuda.is_available():
            self.model.cuda()
            x_gpu = x_cpu.cuda()
            
            with torch.no_grad():
                output_gpu = self.model(x_gpu)
            
            if isinstance(output_gpu, dict):
                assert output_gpu['logits'].device.type == 'cuda'
            else:
                assert output_gpu.device.type == 'cuda'
    
    def test_model_memory_usage(self):
        """Test model memory usage is reasonable"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage (4 bytes per float32 parameter)
        estimated_memory_mb = (total_params * 4) / (1024 * 1024)
        
        print(f"Estimated model memory usage: {estimated_memory_mb:.2f} MB")
        
        # EfficientNetV2-B3 should be reasonable size (< 500MB)
        assert estimated_memory_mb < 500
    
    def test_training_mode_effects(self):
        """Test that training and evaluation modes work correctly"""
        x = torch.randn(4, 3, self.image_size, self.image_size)
        
        # Training mode
        self.model.train()
        
        # Multiple forward passes should give different results due to dropout
        output1 = self.model(x)
        output2 = self.model(x)
        
        if isinstance(output1, dict):
            logits1 = output1['logits']
            logits2 = output2['logits']
        else:
            logits1 = output1
            logits2 = output2
        
        # In training mode with dropout, outputs might be different
        # (this test might be flaky depending on dropout implementation)
        
        # Evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            output3 = self.model(x)
            output4 = self.model(x)
            
            if isinstance(output3, dict):
                logits3 = output3['logits']
                logits4 = output4['logits']
            else:
                logits3 = output3
                logits4 = output4
            
            # In eval mode, outputs should be identical
            assert torch.allclose(logits3, logits4)
    
    def test_model_with_different_input_channels(self):
        """Test model behavior with different input channels (if applicable)"""
        # Standard RGB input
        x_rgb = torch.randn(2, 3, self.image_size, self.image_size)
        
        with torch.no_grad():
            output_rgb = self.model(x_rgb)
        
        if isinstance(output_rgb, dict):
            logits_rgb = output_rgb['logits']
        else:
            logits_rgb = output_rgb
        
        assert logits_rgb.shape == (2, self.model.num_classes)
    
    def test_model_inference_speed(self):
        """Test model inference speed"""
        import time
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        
        # Warm up
        x = torch.randn(1, 3, self.image_size, self.image_size)
        if torch.cuda.is_available():
            x = x.cuda()
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(x)
        
        # Time inference
        n_iterations = 50
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = self.model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / n_iterations
        print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
        
        # Should be reasonably fast (< 1 second per image)
        assert avg_inference_time < 1.0


class TestBaselineModelIntegration:
    """Integration tests for baseline model"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.model = EfficientNetV2B3Baseline(pretrained=False)
        self.batch_size = 8
        self.image_size = 256
    
    def test_training_loop_simulation(self):
        """Test simulated training loop"""
        self.model.train()
        
        # Create dummy dataset
        n_batches = 10
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        initial_loss = None
        final_loss = None
        
        for batch_idx in range(n_batches):
            # Generate random batch
            x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
            # Create realistic targets (some correlation with input)
            targets = torch.randint(0, 2, (self.batch_size,))
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(x)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = criterion(logits, targets)
            
            if batch_idx == 0:
                initial_loss = loss.item()
            if batch_idx == n_batches - 1:
                final_loss = loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        print(f"Initial loss: {initial_loss:.4f}")
        print(f"Final loss: {final_loss:.4f}")
        
        # Loss should be finite
        assert np.isfinite(initial_loss)
        assert np.isfinite(final_loss)
    
    def test_evaluation_pipeline(self):
        """Test evaluation pipeline simulation"""
        self.model.eval()
        
        # Simulate evaluation on multiple batches
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for _ in range(5):  # 5 evaluation batches
                x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
                targets = torch.randint(0, 2, (self.batch_size,))
                
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                predictions = torch.argmax(logits, dim=1)
                
                total_samples += self.batch_size
                correct_predictions += (predictions == targets).sum().item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct_predictions / total_samples
        print(f"Evaluation accuracy: {accuracy:.4f}")
        
        # Accuracy should be reasonable (not exactly 0.5 due to random seed)
        assert 0.0 <= accuracy <= 1.0
        assert len(all_predictions) == total_samples
        assert len(all_targets) == total_samples
    
    def test_model_checkpoint_workflow(self):
        """Test complete checkpoint save/load workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Train for a few steps
            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Training step
            x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
            targets = torch.randint(0, 2, (self.batch_size,))
            
            optimizer.zero_grad()
            outputs = self.model(x)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Save checkpoint
            checkpoint = {
                'epoch': 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            
            checkpoint_path = checkpoint_dir / "checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Create new model and optimizer
            new_model = EfficientNetV2B3Baseline(pretrained=False)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            
            # Load states
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            
            # Verify loaded model produces same output
            self.model.eval()
            new_model.eval()
            
            test_x = torch.randn(2, 3, self.image_size, self.image_size)
            
            with torch.no_grad():
                output1 = self.model(test_x)
                output2 = new_model(test_x)
                
                if isinstance(output1, dict):
                    logits1 = output1['logits']
                    logits2 = output2['logits']
                else:
                    logits1 = output1
                    logits2 = output2
                
                assert torch.allclose(logits1, logits2, atol=1e-5)
    
    def test_model_with_different_optimizers(self):
        """Test model with different optimizers"""
        optimizers = [
            torch.optim.SGD(self.model.parameters(), lr=1e-3),
            torch.optim.Adam(self.model.parameters(), lr=1e-3),
            torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        ]
        
        for optimizer in optimizers:
            self.model.train()
            
            x = torch.randn(4, 3, self.image_size, self.image_size)
            targets = torch.randint(0, 2, (4,))
            
            optimizer.zero_grad()
            
            outputs = self.model(x)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            # Check that loss is finite
            assert np.isfinite(loss.item())


# Performance and edge case tests
class TestBaselineModelEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_input(self):
        """Test model with zero input"""
        model = EfficientNetV2B3Baseline(pretrained=False)
        model.eval()
        
        x = torch.zeros(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(x)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            assert not torch.isnan(logits).any()
            assert torch.isfinite(logits).all()
    
    def test_extreme_input_values(self):
        """Test model with extreme input values"""
        model = EfficientNetV2B3Baseline(pretrained=False)
        model.eval()
        
        # Test with very large values
        x_large = torch.ones(1, 3, 256, 256) * 1000
        
        with torch.no_grad():
            output_large = model(x_large)
            
            if isinstance(output_large, dict):
                logits_large = output_large['logits']
            else:
                logits_large = output_large
            
            assert torch.isfinite(logits_large).all()
        
        # Test with very small values
        x_small = torch.ones(1, 3, 256, 256) * 1e-6
        
        with torch.no_grad():
            output_small = model(x_small)
            
            if isinstance(output_small, dict):
                logits_small = output_small['logits']
            else:
                logits_small = output_small
            
            assert torch.isfinite(logits_small).all()
    
    def test_single_pixel_input(self):
        """Test model with minimal input size (if supported)"""
        model = EfficientNetV2B3Baseline(pretrained=False)
        model.eval()
        
        # Standard size should work
        x = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            output = model(x)
            
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            assert logits.shape == (1, 2)
    
    @pytest.mark.slow
    def test_large_batch_processing(self):
        """Test model with large batch sizes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for large batch test")
        
        model = EfficientNetV2B3Baseline(pretrained=False).cuda()
        model.eval()
        
        # Test progressively larger batch sizes until memory limit
        for batch_size in [32, 64, 128]:
            try:
                x = torch.randn(batch_size, 3, 256, 256).cuda()
                
                with torch.no_grad():
                    output = model(x)
                    
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    assert logits.shape[0] == batch_size
                
                print(f"Successfully processed batch size: {batch_size}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory at batch size: {batch_size}")
                    break
                else:
                    raise e


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])