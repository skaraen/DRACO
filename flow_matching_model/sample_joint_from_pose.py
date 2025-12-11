#!/usr/bin/env python3
"""
Sample joint angles given pose, using a flow-matching model trained by train_fl_pose.py.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import torch

# Add flow_matching directory to path (contains setup.py)
FM_ROOT = Path(__file__).resolve().parent.parent / "flow_matching"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

# Add current directory to path for train_fl_pose import
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from train_fl_pose import (
    ConditionedVelocityNet,
    VelocityModelWrapper,
    sample as fm_sample,
)
from flow_matching.solver import ODESolver


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample joint angles from pose using trained flow-matching model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="flow_matching_model/checkpoint_epoch_1000.pt",
        help="Path to trained checkpoint (.pt) from train_fl_pose.py",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--pose",
        type=float,
        nargs=7,
        help="Single conditioning pose as 7 floats (will create batch_size=1)",
    )
    parser.add_argument(
        "--pose_npy",
        type=str,
        default=None,
        help="Path to .npy or .npz file containing poses of shape (N, 7). For .npz files, expects key 'poses'.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of ODE steps used in sampling",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default=None,
        help="Optional path to save sampled joint angles as .npy",
    )
    return parser.parse_args()


def load_poses(args, device: torch.device) -> torch.Tensor:
    """
    Load poses from either --pose_npy file or --pose argument.
    Supports both single pose (shape: (7,)) and multiple poses (shape: (N, 7)).
    Supports both .npy and .npz files. For .npz files, expects key 'poses'.
    Returns tensor of shape (batch_size, 7) where batch_size >= 1.
    """
    if args.pose_npy is not None:
        file_path = Path(args.pose_npy)
        
        # Handle .npz files
        if file_path.suffix == '.npz':
            npz_data = np.load(args.pose_npy)
            if 'poses' in npz_data:
                poses_np = npz_data['poses']
            else:
                # If no 'poses' key, try to get the first array
                keys = list(npz_data.keys())
                if len(keys) == 0:
                    raise ValueError(f"NPZ file {args.pose_npy} contains no arrays")
                if len(keys) > 1:
                    raise ValueError(f"NPZ file {args.pose_npy} contains multiple arrays. Please use 'poses' key or provide .npy file.")
                poses_np = npz_data[keys[0]]
        else:
            # Handle .npy files
            poses_np = np.load(args.pose_npy)
        
        if poses_np.ndim == 1:
            poses_np = poses_np.reshape(1, -1)  # Single pose: (7,) -> (1, 7)
        # If ndim == 2, keep as is: (N, 7) for multiple poses
        if poses_np.shape[1] != 7:
            raise ValueError(f"Expected pose dimension 7, got shape {poses_np.shape}")
        poses = torch.from_numpy(poses_np).float().to(device)
    elif args.pose is not None:
        if len(args.pose) != 7:
            raise ValueError("You must provide exactly 7 numbers for --pose")
        poses = torch.tensor([args.pose], dtype=torch.float32, device=device)
    else:
        raise ValueError("Please provide either --pose or --pose_npy")
    return poses

def process_pose_directory(pose_dir: Path, args, device: torch.device, model, solver, joint_min: torch.Tensor, joint_max: torch.Tensor,):
    if not pose_dir.is_dir():
        raise NotADirectoryError(f"{pose_dir} is not a directory")

    pose_files = sorted(pose_dir.glob("*_rposes.npz"))
    if not pose_files:
        raise FileNotFoundError(f"No *_rposes.npz files found in directory: {pose_dir}")

    if args.out_npy is not None:
        out_dir = Path(args.out_npy)
    else:
        out_dir = pose_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(pose_files)} pose files in {pose_dir}")
    print(f"Saving joint files to {out_dir}")

    for pose_file in pose_files:
        print(f"\nLoading poses from: {pose_file}")

        npz_data = np.load(pose_file)
        if "poses" in npz_data:
            poses_np = npz_data["poses"]
        else:
            keys = list(npz_data.keys())
            if len(keys) == 0:
                raise ValueError(f"NPZ file {pose_file} contains no arrays")
            if len(keys) > 1:
                raise ValueError(
                    f"NPZ file {pose_file} contains multiple arrays. "
                    f"Please use a 'poses' key or a single-array file."
                )
            poses_np = npz_data[keys[0]]

        if poses_np.ndim == 1:
            poses_np = poses_np.reshape(1, -1)  # (7,) -> (1, 7)

        if poses_np.shape[1] != 7:
            raise ValueError(f"Expected pose dimension 7, got shape {poses_np.shape}")

        poses = torch.from_numpy(poses_np).float().to(device)
        print(f"  Poses shape: {poses.shape}")

        with torch.no_grad():
            joints = fm_sample(
                model=model,
                pose=poses,
                solver=solver,
                device=device,
                joint_min=joint_min,
                joint_max=joint_max,
                num_steps=args.num_steps,
            )

        joints_np = joints.cpu().numpy()

        stem = pose_file.stem
        if stem.endswith("_rposes"):
            out_stem = stem[:-len("_rposes")] + "_joints"
        else:
            out_stem = stem + "_joints"

        out_path = out_dir / f"{out_stem}.npy"
        np.save(out_path, joints_np)
        print(f"  Saved joints with shape {joints_np.shape} to: {out_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint to CPU first to avoid GPU memory issues
    print("Loading checkpoint from CPU...")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Build model; hyperparameters must match training-time values.
    # Here we assume default dims (pose_dim=7, joint_dim=4) as in train_fl_pose.py.
    print("Building model...")
    model = ConditionedVelocityNet(
        pose_dim=7,
        joint_dim=4,
        d_model=512,
        num_layers=6,
        d_ff=2048,
        num_heads=8,
        dropout=0.1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    # Move model to device after loading state dict
    model = model.to(device)
    model.eval()

    # Build solver with wrapped model
    wrapped_model = VelocityModelWrapper(model)
    solver = ODESolver(velocity_model=wrapped_model)

    # Load normalization stats for joints
    if "joint_min" not in ckpt or "joint_max" not in ckpt:
        raise KeyError("Checkpoint does not contain 'joint_min'/'joint_max' statistics")
    joint_min = ckpt["joint_min"].to(device).view(1, -1)
    joint_max = ckpt["joint_max"].to(device).view(1, -1)

    if args.pose_npy is not None:
        pose_path = Path(args.pose_npy)
        if pose_path.is_dir():
            print(f"--pose_npy is a directory, processing all *_rposes.npz in: {pose_path}")
            process_pose_directory(
                pose_dir=pose_path,
                args=args,
                device=device,
                model=model,
                solver=solver,
                joint_min=joint_min,
                joint_max=joint_max,
            )
            return

    # Otherwise: existing single-file / single-pose behavior
    poses = load_poses(args, device)
    print(f"Loaded poses with shape: {poses.shape}")

    # Sample joint angles
    with torch.no_grad():
        joints = fm_sample(
            model=model,
            pose=poses,
            solver=solver,
            device=device,
            joint_min=joint_min,
            joint_max=joint_max,
            num_steps=args.num_steps,
        )

    joints_np = joints.cpu().numpy()
    print("Sampled joint angles (numpy):")
    print(joints_np)

    if args.out_npy is not None:
        out_path = Path(args.out_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, joints_np)
        print(f"Saved sampled joint angles to: {out_path}")


if __name__ == "__main__":
    main()


