#!/usr/bin/env python3
"""
Debug script to analyze the problematic model file
"""

import torch
import json
from pathlib import Path
from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast


def debug_model(model_path: str):
    """Debug a specific model file"""

    print(f"🔍 Debugging model: {model_path}")
    print("=" * 60)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return

    # Check file size
    file_size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"📁 File size: {file_size_mb:.1f} MB")

    try:
        # Load the checkpoint
        print("📦 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')

        # Analyze checkpoint structure
        print(f"🔑 Checkpoint keys: {list(checkpoint.keys())}")

        # Extract and display config
        config = checkpoint.get('config', {})
        print(f"\n⚙️  Model Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        # Extract training stats
        training_stats = checkpoint.get('training_stats', {})
        print(f"\n📊 Training Statistics:")
        for key, value in training_stats.items():
            print(f"   {key}: {value}")

        # Determine expected architecture from config
        hidden_dim = config.get('hidden_dim', 64)
        use_fast_ode = config.get('use_fast_ode', True)

        print(f"\n🏗️  Expected Architecture:")
        print(f"   use_fast_ode: {use_fast_ode}")
        print(f"   hidden_dim: {hidden_dim}")
        print(f"   Model class: {'EvoformerODEFuncFast' if use_fast_ode else 'EvoformerODEFunc'}")

        # Try to create the model architecture
        c_m = 256  # MSA embedding dimension
        c_z = 128  # Pair embedding dimension

        if use_fast_ode:
            model = EvoformerODEFuncFast(c_m, c_z, hidden_dim)
        else:
            model = EvoformerODEFunc(c_m, c_z, hidden_dim)

        print(f"✅ Model architecture created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")

        # Try to load the state dict
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model state loaded successfully")

            # Test model with dummy input
            print(f"\n🧪 Testing model with dummy input...")
            test_model_inference(model, hidden_dim)

        except Exception as e:
            print(f"❌ Failed to load model state: {e}")

            # Analyze state dict
            state_dict = checkpoint.get('model_state_dict', {})
            print(f"\n🔍 State dict analysis:")
            print(f"   State dict keys: {len(state_dict)} parameters")

            # Show parameter shapes
            for key, tensor in list(state_dict.items())[:10]:  # First 10 params
                print(f"     {key}: {tensor.shape}")
            if len(state_dict) > 10:
                print(f"     ... and {len(state_dict) - 10} more parameters")

            # Check for dimension mismatches
            model_state = model.state_dict()
            print(f"\n🔍 Parameter comparison:")
            mismatches = []
            for key in model_state.keys():
                if key in state_dict:
                    if model_state[key].shape != state_dict[key].shape:
                        mismatches.append((key, model_state[key].shape, state_dict[key].shape))
                else:
                    print(f"   Missing in checkpoint: {key}")

            if mismatches:
                print(f"   Shape mismatches found:")
                for key, expected, actual in mismatches:
                    print(f"     {key}: expected {expected}, got {actual}")
            else:
                print(f"   No shape mismatches found")

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"   This suggests the file is corrupted or incomplete")


def test_model_inference(model, hidden_dim):
    """Test model with dummy input to verify it works"""

    # Create dummy input tensors
    batch_size = 1
    n_seq = 32
    n_res = 64
    c_m = 256
    c_z = 128

    # Dummy MSA and pair representations
    m_dummy = torch.randn(n_seq, n_res, c_m)
    z_dummy = torch.randn(n_res, n_res, c_z)

    try:
        model.eval()
        with torch.no_grad():
            # Test the ODE function
            dm_dt, dz_dt = model(0.0, (m_dummy, z_dummy))

            print(f"   ✅ Model inference successful")
            print(f"   dm_dt shape: {dm_dt.shape}")
            print(f"   dz_dt shape: {dz_dt.shape}")
            print(f"   Output contains NaN: {torch.isnan(dm_dt).any() or torch.isnan(dz_dt).any()}")

            return True

    except Exception as e:
        print(f"   ❌ Model inference failed: {e}")
        return False


def compare_models(model1_path: str, model2_path: str):
    """Compare two model files"""

    print(f"\n🔍 Comparing models:")
    print(f"   Model 1: {model1_path}")
    print(f"   Model 2: {model2_path}")
    print("=" * 60)

    for i, path in enumerate([model1_path, model2_path], 1):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            config = checkpoint.get('config', {})

            print(f"\nModel {i} ({Path(path).name}):")
            print(f"   use_fast_ode: {config.get('use_fast_ode')}")
            print(f"   hidden_dim: {config.get('hidden_dim')}")
            print(f"   reduced_cluster_size: {config.get('reduced_cluster_size')}")
            print(f"   integrator: {config.get('integrator')}")
            print(f"   preliminary_training: {config.get('enable_preliminary_training')}")

            training_stats = checkpoint.get('training_stats', {})
            print(f"   final_val_loss: {training_stats.get('final_val_loss')}")
            print(f"   early_stopped: {training_stats.get('early_stopped')}")
            print(f"   total_epochs: {training_stats.get('total_epochs')}")

        except Exception as e:
            print(f"Model {i}: ❌ Failed to load - {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_model.py <model_path>")
        print("  python debug_model.py <model1_path> <model2_path>  # Compare two models")
        print("")
        print("Example:")
        print("  python debug_model.py trained_models/20250615_180436_full_ode_with_prelim_64_final_model.pt")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Debug single model
        debug_model(sys.argv[1])
    else:
        # Compare two models
        debug_model(sys.argv[1])
        debug_model(sys.argv[2])
        compare_models(sys.argv[1], sys.argv[2])