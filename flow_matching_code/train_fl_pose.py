#!/usr/bin/env python3
"""
Train flow matching model to predict pose from joint angles
"""

import os
import sys
from pathlib import Path
import argparse
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Add flow_matching to path if needed
FM_ROOT = Path(__file__).resolve().parent / "flow_matching"
if FM_ROOT.exists() and str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

# Add current directory to path for transformer (in flow_matching folder)
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler
    from flow_matching.solver import ODESolver
    from flow_matching.utils import ModelWrapper as FMModelWrapper
except ImportError:
    # If flow_matching is not in path, try to add it
    FM_ROOT = Path(__file__).resolve().parent / "flow_matching"
    if FM_ROOT.exists():
        sys.path.insert(0, str(FM_ROOT))
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler
    from flow_matching.solver import ODESolver
    from flow_matching.utils import ModelWrapper as FMModelWrapper

# Import transformer from local flow_matching folder
from transformer import make_transformer


class JointToPoseDataset(Dataset):
    """Dataset for pose -> joint-angles mapping"""
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to .pt file containing list of (joint, pose) tuples
        """
        self.data = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Stored order is (joint, pose); we return (pose, joint) for pose->joint mapping
        joint, pose = self.data[idx]
        return pose, joint


def compute_joint_minmax_stats(dataset: JointToPoseDataset):
    """
    Compute per-dimension min/max over all joint-angle vectors in the dataset.
    Returns:
        joint_min, joint_max: tensors of shape (joint_dim,) on CPU.
    """
    # data is list[(joint, pose)], where joint is 1D tensor of length joint_dim
    joints = torch.stack([j for (j, _) in dataset.data], dim=0).float()  # (N, joint_dim)

    joint_min = torch.amin(joints, dim=0)
    joint_max = torch.amax(joints, dim=0)

    # Fallback if degenerate ranges
    joint_min = torch.where(torch.isfinite(joint_min), joint_min, torch.zeros_like(joint_min))
    joint_max = torch.where(torch.isfinite(joint_max), joint_max, torch.ones_like(joint_max))

    degenerate = (joint_max - joint_min) < 1e-6
    joint_max[degenerate] = joint_min[degenerate] + 1.0
    return joint_min, joint_max


class ConditionedVelocityNet(nn.Module):
    """Neural network that predicts joint-angle velocity given joint state, time, and pose condition using transformer"""
    
    def __init__(self, pose_dim=7, joint_dim=4, d_model=512, num_layers=6, d_ff=2048, num_heads=8, dropout=0.1):
        """
        Args:
            pose_dim: Dimension of conditioning pose (default: 7)
            joint_dim: Dimension of joint angles / state x (default: 4)
            d_model: Model dimension (default: 512)
            num_layers: Number of transformer layers (default: 6)
            d_ff: Feed-forward dimension (default: 2048)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.pose_dim = pose_dim
        self.joint_dim = joint_dim
        self.d_model = d_model
        
        # Projections to d_model
        self.joint_proj = nn.Linear(joint_dim, d_model)  # state x_t (joints)
        self.pose_proj = nn.Linear(pose_dim, d_model)   # conditioning pose
        
        # Time embedding using MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer encoder
        self.transformer = make_transformer(
            N=num_layers,
            d_model=d_model,
            d_ff=d_ff,
            h=num_heads,
            dropout=dropout
        )
        
        # Output projection to joint-angle velocity in joint space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, joint_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Joint-angle state tensor, shape (batch_size, joint_dim)
            t: Time tensor, shape (batch_size,) or (batch_size, 1)
            pose: Conditioning pose tensor, shape (batch_size, pose_dim)
        
        Returns:
            Predicted joint-angle velocity, shape (batch_size, joint_dim)
        """
        batch_size = x.shape[0]
        
        # Ensure t is (batch_size, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 0:
            t = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, 1)
        
        # Project to d_model
        joint_token = self.joint_proj(x).unsqueeze(1)   # (batch_size, 1, d_model)
        pose_token = self.pose_proj(pose).unsqueeze(1)  # (batch_size, 1, d_model)
        time_token = self.time_mlp(t).unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Concat tokens instead of adding
        tokens = torch.cat([joint_token, pose_token, time_token], dim=1)  # (batch_size, 3, d_model)
        
        # Pass through transformer
        out = self.transformer(tokens, mask=None)  # (batch_size, 3, d_model)
        
        # Take only the joint token output (first token)
        joint_out = out[:, 0, :]  # (batch_size, d_model)
        
        # Project to velocity in joint space
        velocity = self.output_proj(joint_out)  # (batch_size, joint_dim)
        
        return velocity


class VelocityModelWrapper(FMModelWrapper):
    """Wrapper to make model compatible with flow_matching solver"""
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Joint-angle state tensor, shape (batch_size, joint_dim)
            t: Time tensor, shape (batch_size,)
            **kwargs: Additional arguments (should contain 'pose')
        
        Returns:
            Predicted joint-angle velocity, shape (batch_size, joint_dim)
        """
        pose = kwargs.get('pose')
        if pose is None:
            raise ValueError("pose condition must be provided in kwargs")
        return self.model(x=x, t=t, pose=pose)


def train_epoch(model, dataloader, path, optimizer, device, epoch, joint_min, joint_max):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (pose, joint) in enumerate(dataloader):
        pose = pose.to(device)
        joint = joint.to(device)
        batch_size = pose.shape[0]
        
        # Sample random time
        t = torch.rand(batch_size, device=device)
        
        # Normalize joint angles to [-1, 1] per-dimension
        scale = (joint_max - joint_min).clamp_min(1e-6)
        x_1 = (joint - joint_min) / scale * 2.0 - 1.0
        x_1 = x_1.clamp(-1.0, 1.0)

        # Sample random noise for x_0
        x_0 = torch.randn_like(x_1)
        
        # Sample from path: x_t ~ p_t(x_t | x_0, x_1)
        path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t  # Ground truth velocity
        
        # Predict velocity
        pred_dx_t = model(x=x_t, t=t, pose=pose)
        
        # Compute loss
        loss = F.mse_loss(pred_dx_t, dx_t)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, path, device, joint_min, joint_max):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for pose, joint in dataloader:
        pose = pose.to(device)
        joint = joint.to(device)
        batch_size = pose.shape[0]
        
        # Sample random time
        t = torch.rand(batch_size, device=device)
        
        # Normalize joint angles to [-1, 1] per-dimension
        scale = (joint_max - joint_min).clamp_min(1e-6)
        x_1 = (joint - joint_min) / scale * 2.0 - 1.0
        x_1 = x_1.clamp(-1.0, 1.0)

        # Sample random noise for x_0
        x_0 = torch.randn_like(x_1)
        
        # Sample from path
        path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t = path_sample.dx_t
        
        # Predict velocity
        pred_dx_t = model(x=x_t, t=t, pose=pose)
        
        # Compute loss
        loss = F.mse_loss(pred_dx_t, dx_t)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def sample(model, pose, solver, device, joint_min, joint_max, num_steps=100):
    """
    Sample joint angles from pose using flow matching
    
    Args:
        model: Trained velocity model (ConditionedVelocityNet)
        pose: Conditioning poses, shape (batch_size, pose_dim)
        solver: ODE solver
        device: Device to run on
        num_steps: Number of ODE steps
    
    Returns:
        Sampled joint angles in original (denormalized) joint space, shape (batch_size, joint_dim)
    """
    model.eval()
    pose = pose.to(device)
    batch_size = pose.shape[0]
    
    # Start from random noise in joint space
    x_init = torch.randn(batch_size, model.joint_dim, device=device)
    
    # Create time grid from 0 to 1
    time_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    
    # Solve ODE from t=0 to t=1 (in normalized pose space)
    x_1 = solver.sample(
        x_init=x_init,
        step_size=None,  # Use time_grid for discretization
        time_grid=time_grid,
        pose=pose  # Pass pose as extra conditioning
    )

    # Denormalize from [-1, 1] back to original joint-angle space
    # joint_min/joint_max are (1, joint_dim); broadcast to batch
    scale = (joint_max - joint_min).clamp_min(1e-6)
    x_1 = x_1.clamp(-1.0, 1.0)
    x_1_denorm = (x_1 + 1.0) * 0.5 * scale + joint_min
    
    return x_1_denorm


def main():
    parser = argparse.ArgumentParser(description="Train flow matching model for joint to pose prediction")
    parser.add_argument("--train_data", type=str, default="data/joint_to_pose_train.pt",
                       help="Path to training data")
    parser.add_argument("--test_data", type=str, default="data/joint_to_pose_test.pt",
                       help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=2048,
                       help="Feed-forward dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--save_dir", type=str, default="output/model_training",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_freq", type=int, default=50,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--eval_freq", type=int, default=1,
                       help="Evaluate every N epochs")
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = JointToPoseDataset(args.train_data)
    test_dataset = JointToPoseDataset(args.test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Compute joint-angle min/max for normalization (on training set)
    joint_min_cpu, joint_max_cpu = compute_joint_minmax_stats(train_dataset)
    # Prepare broadcastable tensors on device
    joint_min = joint_min_cpu.to(device).view(1, -1)
    joint_max = joint_max_cpu.to(device).view(1, -1)
    
    # Create model
    model = ConditionedVelocityNet(
        pose_dim=7,
        joint_dim=4,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create flow matching path
    scheduler = CondOTScheduler()
    path = AffineProbPath(scheduler=scheduler)
    
    # Create solver for sampling
    wrapped_model = VelocityModelWrapper(model)
    solver = ODESolver(velocity_model=wrapped_model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    test_epochs = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, path, optimizer, device, epoch, joint_min, joint_max)
        print(f"Train Loss: {train_loss:.6f}")
        train_losses.append(train_loss)
        
        # Evaluate
        if epoch % args.eval_freq == 0:
            test_loss = evaluate(model, test_loader, path, device, joint_min, joint_max)
            print(f"Test Loss: {test_loss:.6f}")
            test_losses.append(test_loss)
            test_epochs.append(epoch)
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'joint_min': joint_min_cpu,
                    'joint_max': joint_max_cpu,
                }
                torch.save(checkpoint, save_dir / "best_model.pt")
                print(f"Saved best model (test loss: {test_loss:.6f})")
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'joint_min': joint_min_cpu,
                'joint_max': joint_max_cpu,
            }
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"Saved checkpoint at epoch {epoch}")
    
    print("\nTraining completed!")
    print(f"Best test loss: {best_test_loss:.6f}")

    # Save loss curves
    epochs = np.arange(1, args.epochs + 1)
    train_losses_np = np.array(train_losses, dtype=np.float32)
    test_epochs_np = np.array(test_epochs, dtype=np.int32)
    test_losses_np = np.array(test_losses, dtype=np.float32)

    # Save raw data
    np.savez(save_dir / "loss_curves.npz",
             epochs=epochs,
             train_losses=train_losses_np,
             test_epochs=test_epochs_np,
             test_losses=test_losses_np)

    # Plot and save PNG
    plt.figure()
    plt.plot(epochs, train_losses_np, label="Train Loss")
    if len(test_epochs_np) > 0:
        plt.plot(test_epochs_np, test_losses_np, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Flow Matching Train/Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curves.png")
    plt.close()


if __name__ == "__main__":
    main()

