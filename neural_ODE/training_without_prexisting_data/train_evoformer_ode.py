import os
import gc
import torch
import argparse
import sys
import subprocess
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc, EvoformerODEFuncFast
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import json
import time
from pathlib import Path


def get_project_root():
    """Get the path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir


# === Parse command line arguments ===
parser = argparse.ArgumentParser(description='Train Evoformer ODE - Memory Optimized with Proper Splits')

# Data directory options
parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')

# Model options
parser.add_argument('--use_fast_ode', action='store_true', default=False, help='Use the faster implementation')
parser.add_argument('--no-use_fast_ode', dest='use_fast_ode', action='store_false', help='Disable fast ODE')

# Memory and model size options
parser.add_argument('--memory_split_size', type=int, default=128, help='Memory split size (MB)')
parser.add_argument('--reduced_cluster_size', type=int, default=32, help='Maximum cluster size')
parser.add_argument('--reduced_hidden_dim', type=int, default=32, help='Hidden dimension size')

# Training options
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--num_time_points', type=int, default=5, help='Number of integration time points')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='euler', help='ODE integrator')
parser.add_argument('--gradient_accumulation', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--chunk_size', type=int, default=0, help='Chunk size')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')

# Optimization flags
parser.add_argument('--use_amp', action='store_true', default=False, help='Use Automatic Mixed Precision')
parser.add_argument('--no-use_amp', dest='use_amp', action='store_false', help='Disable AMP')
parser.add_argument('--use_checkpoint', action='store_true', default=False, help='Use gradient checkpointing')
parser.add_argument('--no-use_checkpoint', dest='use_checkpoint', action='store_false', help='Disable checkpointing')
parser.add_argument('--monitor_memory', action='store_true', default=False, help='Monitor memory usage')
parser.add_argument('--no-monitor_memory', dest='monitor_memory', action='store_false',
                    help='Disable memory monitoring')
parser.add_argument('--clean_memory', action='store_true', default=False, help='Clean memory aggressively')
parser.add_argument('--no-clean_memory', dest='clean_memory', action='store_false', help='Disable memory cleaning')
parser.add_argument('--reduced_precision_integration', action='store_true', default=False, help='Use reduced precision')
parser.add_argument('--no-reduced_precision_integration', dest='reduced_precision_integration', action='store_false',
                    help='Disable reduced precision')

# Device options
parser.add_argument('--cpu_only', action='store_true', default=False, help='Force CPU-only mode')
parser.add_argument('--no-cpu-only', dest='cpu_only', action='store_false', help='Disable CPU-only mode')

# Test options
parser.add_argument('--test-protein', type=str, default=None, help='Test specific protein')

# Additional options
parser.add_argument('--max_iterations', type=int, default=10, help='Maximum iterations')
parser.add_argument('--config_preset', type=str, default='model_1_ptm', help='Config preset')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')

# Early stopping options
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

args = parser.parse_args()

# === Configuration ===
PROJECT_ROOT = get_project_root()

if args.data_dir is None:
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
else:
    DATA_DIR = args.data_dir

if args.output_dir is None:
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
else:
    OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device configuration
if args.cpu_only:
    device = "cpu"
    args.use_amp = False
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        args.use_amp = False

print(f"🚀 Evoformer ODE Training - Memory Optimized with Proper Splits")
print(f"📁 Data directory: {DATA_DIR}")
print(f"💾 Output directory: {OUTPUT_DIR}")
print(f"💻 Device: {device}")
print(f"⚙️  Configuration:")
print(f"   Learning rate: {args.learning_rate}")
print(f"   Epochs: {args.epochs}")
print(f"   Time points: {args.num_time_points}")
print(f"   Cluster size: {args.reduced_cluster_size}")
print(f"   Hidden dim: {args.reduced_hidden_dim}")
print(f"   Use fast ODE: {args.use_fast_ode}")
print(f"   Use AMP: {args.use_amp}")
print(f"   Early stopping patience: {args.patience}")

# Model configuration
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = args.reduced_hidden_dim
learning_rate = args.learning_rate
epochs = args.epochs

# Setup gradient scaler for mixed precision
scaler = GradScaler(enabled=args.use_amp and device == "cuda")


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_stats(label=""):
    """Print memory statistics if monitoring is enabled"""
    if not args.monitor_memory:
        return

    if device == "cpu":
        print(f"=== Memory Stats {label} (CPU mode) ===")
        return

    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2

        # Calculate available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        available = total_memory - allocated

        print(f"=== Memory Stats {label} ===")
        print(f"Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
        print(f"Available: {available:.1f} MB, Max Used: {max_allocated:.1f} MB")

        # Warning if memory is getting low
        if available < 2000:  # Less than 2GB available
            print(f"⚠️  WARNING: Low GPU memory! Only {available:.1f} MB available")

    except Exception as e:
        print(f"Memory stats error: {e}")


def get_dataset(input_dir, split='training'):
    """Get all protein folder names that contain evoformer blocks data from specified split"""
    datasets = []
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"❌ Data directory not found: {input_dir}")
        return datasets

    # Handle different directory structures
    if (input_path / "training").exists() or (input_path / "validation").exists():
        # Mini-data structure with splits - ONLY use specified split
        split_dir = input_path / split / 'blocks'
        if split_dir.exists():
            print(f"📁 Loading proteins from {split} split: {split_dir}")
            for protein_dir in split_dir.iterdir():
                if (protein_dir.is_dir() and
                        protein_dir.name.endswith('_evoformer_blocks') and
                        (protein_dir / 'recycle_0').exists()):
                    protein_id = protein_dir.name.replace('_evoformer_blocks', '')
                    datasets.append(protein_id)
                    print(f"  ✅ Found: {protein_id}")
        else:
            print(f"⚠️  Split directory not found: {split_dir}")
    else:
        # Standard structure - all proteins in one directory (fallback)
        print(f"📁 Using flat structure (no splits): {input_path}")
        for protein_dir in input_path.iterdir():
            if (protein_dir.is_dir() and
                    protein_dir.name.endswith('_evoformer_blocks') and
                    (protein_dir / 'recycle_0').exists()):
                protein_id = protein_dir.name.replace('_evoformer_blocks', '')
                datasets.append(protein_id)

    print(f"📊 Total proteins loaded from {split}: {len(datasets)}")
    return datasets


def find_protein_data(protein_id, data_dir):
    """Find protein data in various possible locations"""
    data_path = Path(data_dir)

    possible_locations = [
        data_path / f"{protein_id}_evoformer_blocks" / "recycle_0",
        data_path / "training" / "blocks" / f"{protein_id}_evoformer_blocks" / "recycle_0",
        data_path / "validation" / "blocks" / f"{protein_id}_evoformer_blocks" / "recycle_0",
        data_path / "testing" / "blocks" / f"{protein_id}_evoformer_blocks" / "recycle_0",
    ]

    for location in possible_locations:
        if location.exists():
            return location

    return None


def load_tensor_with_cleanup(path, device, cluster_size=None):
    """Load tensor with immediate memory optimization"""
    tensor = torch.load(path, map_location=device)

    # Remove batch dimension if present
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Apply cluster size reduction if specified (only for MSA dimension)
    if cluster_size is not None and tensor.dim() == 3 and tensor.shape[0] > cluster_size:
        tensor = tensor[:cluster_size]

    return tensor


def load_initial_input(protein_id):
    """Load the initial M and Z tensors for block 0"""
    protein_dir = find_protein_data(protein_id, DATA_DIR)

    if protein_dir is None:
        raise FileNotFoundError(f"Data not found for {protein_id}")

    m_path = protein_dir / "m_block_0.pt"
    z_path = protein_dir / "z_block_0.pt"

    if not m_path.exists() or not z_path.exists():
        raise FileNotFoundError(f"Initial block files not found for {protein_id}")

    # Load tensors with immediate cleanup
    m_init = load_tensor_with_cleanup(m_path, device, args.reduced_cluster_size)
    z_init = load_tensor_with_cleanup(z_path, device)

    return m_init, z_init


def balanced_loss_fn(pred_m, target_m, pred_z, target_z, msa_weight=1.0, pair_weight=0.1):
    """Balanced loss function that handles size mismatches"""
    # Handle size mismatches due to sequence length truncation
    pred_seq_len = pred_m.shape[1]
    target_seq_len = target_m.shape[1]

    if pred_seq_len != target_seq_len:
        min_len = min(pred_seq_len, target_seq_len)
        pred_m = pred_m[:, :min_len, :]
        target_m = target_m[:, :min_len, :]
        pred_z = pred_z[:min_len, :min_len, :]
        target_z = target_z[:min_len, :min_len, :]

    msa_loss = F.mse_loss(pred_m, target_m)
    pair_loss = F.mse_loss(pred_z, target_z)
    weighted_loss = msa_weight * msa_loss + pair_weight * pair_loss

    return weighted_loss, msa_loss.item(), pair_loss.item()


# Initialize model
if args.use_fast_ode:
    print("Using FAST EvoformerODEFunc implementation")
    ode_func = EvoformerODEFuncFast(c_m, c_z, hidden_dim).to(device)
else:
    print("Using standard EvoformerODEFunc implementation")
    ode_func = EvoformerODEFunc(c_m, c_z, hidden_dim).to(device)

optimizer = torch.optim.Adam(ode_func.parameters(), lr=learning_rate)


class EvoformerIterationManager:
    """Manages calling evoformer_iter to generate next time steps on-demand with automatic cleanup"""

    def __init__(self, project_root, config_preset='model_1_ptm', device='cuda:0'):
        self.project_root = Path(project_root)
        self.config_preset = config_preset
        self.device = device
        self.generated_files = {}  # Track files we've generated for cleanup

        # Find the evoformer iteration script
        possible_scripts = [
            self.project_root.parent / "evoformer_iter" / "run_evoformer_iter.py",
            self.project_root / "evoformer_iter" / "run_evoformer_iter.py",
            self.project_root / "run_evoformer_iter.py",
            self.project_root.parent / "evoformer_iter_script.sh",
            self.project_root / "evoformer_iter_script.sh",
        ]

        self.iter_script = None
        for script_path in possible_scripts:
            if script_path.exists():
                self.iter_script = script_path
                break

        if self.iter_script is None:
            raise FileNotFoundError("Could not find evoformer iteration script")

    def generate_next_iteration(self, protein_id: str, current_idx: int) -> tuple:
        """Generate the next Evoformer iteration and return the tensors"""
        protein_dir = find_protein_data(protein_id, DATA_DIR)
        if protein_dir is None:
            raise FileNotFoundError(f"Protein data not found for {protein_id}")

        # Check if current files exist
        m_current = protein_dir / f"m_block_{current_idx}.pt"
        z_current = protein_dir / f"z_block_{current_idx}.pt"

        if not (m_current.exists() and z_current.exists()):
            raise FileNotFoundError(f"Current iteration files not found: {m_current}, {z_current}")

        # Check if next iteration already exists
        m_next = protein_dir / f"m_block_{current_idx + 1}.pt"
        z_next = protein_dir / f"z_block_{current_idx + 1}.pt"

        if m_next.exists() and z_next.exists():
            # Load and track for cleanup
            result = self._load_iteration(m_next, z_next)
            self._track_generated_file(protein_id, current_idx + 1, m_next, z_next)
            return result

        # Generate next iteration using evoformer_iter
        success = self._run_evoformer_iter(m_current, z_current, protein_dir)

        if success and m_next.exists() and z_next.exists():
            # Load and track for cleanup
            result = self._load_iteration(m_next, z_next)
            self._track_generated_file(protein_id, current_idx + 1, m_next, z_next)
            return result
        else:
            raise RuntimeError(f"Failed to generate iteration {current_idx + 1}")

    def _load_iteration(self, m_path: Path, z_path: Path) -> tuple:
        """Load m and z tensors from files with memory optimization"""
        m = load_tensor_with_cleanup(m_path, device, args.reduced_cluster_size)
        z = load_tensor_with_cleanup(z_path, device)
        return m, z

    def _track_generated_file(self, protein_id: str, block_idx: int, m_path: Path, z_path: Path):
        """Track generated files for cleanup"""
        if protein_id not in self.generated_files:
            self.generated_files[protein_id] = {}
        self.generated_files[protein_id][block_idx] = (m_path, z_path)

    def cleanup_protein_files(self, protein_id: str, keep_blocks=None, specific_block=None):
        """Clean up generated files for a protein, keeping only specified blocks or deleting a specific block"""
        if keep_blocks is None:
            keep_blocks = [0]  # Always keep the initial block

        if protein_id not in self.generated_files:
            return

        files_to_delete = []

        if specific_block is not None:
            # Delete only a specific block
            if specific_block in self.generated_files[protein_id]:
                m_path, z_path = self.generated_files[protein_id][specific_block]
                files_to_delete.append((specific_block, m_path, z_path))
        else:
            # Delete all blocks except those in keep_blocks
            for block_idx, (m_path, z_path) in self.generated_files[protein_id].items():
                if block_idx not in keep_blocks:
                    files_to_delete.append((block_idx, m_path, z_path))

        # Delete files
        deleted_count = 0
        freed_size = 0
        for block_idx, m_path, z_path in files_to_delete:
            try:
                if m_path.exists():
                    size = m_path.stat().st_size
                    m_path.unlink()
                    freed_size += size
                    deleted_count += 1

                if z_path.exists():
                    size = z_path.stat().st_size
                    z_path.unlink()
                    freed_size += size
                    deleted_count += 1

                # Remove from tracking
                del self.generated_files[protein_id][block_idx]

            except Exception as e:
                print(f"        ⚠️  Error deleting {m_path.name}/{z_path.name}: {e}")

    def cleanup_specific_timestep(self, time_step: int):
        """Clean up files for a specific time step across all proteins"""
        deleted_count = 0
        freed_size = 0

        for protein_id in list(self.generated_files.keys()):
            if time_step in self.generated_files[protein_id]:
                m_path, z_path = self.generated_files[protein_id][time_step]

                try:
                    if m_path.exists():
                        size = m_path.stat().st_size
                        m_path.unlink()
                        freed_size += size
                        deleted_count += 1

                    if z_path.exists():
                        size = z_path.stat().st_size
                        z_path.unlink()
                        freed_size += size
                        deleted_count += 1

                    # Remove from tracking
                    del self.generated_files[protein_id][time_step]

                except Exception as e:
                    pass

        return deleted_count

    def cleanup_all_generated_files(self, keep_blocks=None):
        """Clean up all generated files across all proteins"""
        if keep_blocks is None:
            keep_blocks = [0]  # Always keep the initial block

        for protein_id in list(self.generated_files.keys()):
            self.cleanup_protein_files(protein_id, keep_blocks)

    def _run_evoformer_iter(self, m_path: Path, z_path: Path, output_dir: Path) -> bool:
        """Run the evoformer iteration script with concise error reporting"""
        try:
            if self.iter_script.suffix == '.sh':
                cmd = [
                    'bash', str(self.iter_script),
                    '--m_path', str(m_path),
                    '--z_path', str(z_path),
                    '--output_dir', str(output_dir),
                    '--config_preset', self.config_preset,
                    '--device', self.device
                ]
            else:
                cmd = [
                    sys.executable, str(self.iter_script),
                    '--m_path', str(m_path),
                    '--z_path', str(z_path),
                    '--output_dir', str(output_dir),
                    '--config_preset', self.config_preset,
                    '--device', self.device
                ]

            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{str(self.project_root)}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = str(self.project_root)

            # Run with reduced timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                cwd=str(self.project_root),
                env=env
            )

            # Check expected output files
            current_idx = int(m_path.stem.split('_')[-1])
            next_idx = current_idx + 1
            m_next = output_dir / f"m_block_{next_idx}.pt"
            z_next = output_dir / f"z_block_{next_idx}.pt"

            if result.returncode == 0 and m_next.exists() and z_next.exists():
                return True
            else:
                return False

        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            return False


# Initialize the iteration manager
iter_manager = EvoformerIterationManager(PROJECT_ROOT, args.config_preset, args.device)


def evaluate_step(split='validation'):
    """Evaluation step for validation or testing"""
    dataset = get_dataset(DATA_DIR, split=split)

    if not dataset:
        print(f"❌ No proteins found in {split} data directory!")
        return 0.0

    print(f"🔍 Evaluating on {len(dataset)} proteins from {split.upper()} split")

    ode_func.eval()  # Set model to evaluation mode

    total_loss = 0
    num_evaluations = 0

    with torch.no_grad():  # No gradients needed for evaluation
        # Load initial states for all proteins
        protein_states = {}
        successful_proteins = []

        for protein_id in dataset:
            try:
                m_init, z_init = load_initial_input(protein_id)
                protein_states[protein_id] = (m_init, z_init)
                successful_proteins.append(protein_id)
            except Exception as e:
                print(f"      ❌ {protein_id}: Failed to load - {e}")
                continue

        if not successful_proteins:
            print(f"    ❌ No proteins loaded successfully from {split}!")
            return 0.0

        # Evaluate each time step
        for time_step in range(1, args.num_time_points):
            t_step = torch.tensor([0.0, float(time_step) / (args.num_time_points - 1)], device=device)

            step_loss = 0
            step_comparisons = 0

            for protein_id in successful_proteins:
                try:
                    m_current, z_current = protein_states[protein_id]

                    # Run ODE prediction
                    ode_state = (m_current.clone(), z_current.clone())

                    if args.reduced_precision_integration:
                        pred_trajectory = odeint(
                            ode_func, ode_state, t_step, method=args.integrator, rtol=1e-2, atol=1e-2
                        )
                    else:
                        pred_trajectory = odeint(
                            ode_func, ode_state, t_step, method=args.integrator
                        )

                    pred_m = pred_trajectory[0][1]
                    pred_z = pred_trajectory[1][1]

                    # Generate ground truth
                    target_m, target_z = iter_manager.generate_next_iteration(protein_id, time_step - 1)

                    # Compute loss
                    protein_loss, _, _ = balanced_loss_fn(pred_m, target_m, pred_z, target_z)
                    step_loss += protein_loss.item()
                    step_comparisons += 1

                    # Update state for next time step
                    protein_states[protein_id] = (pred_m.detach().clone(), pred_z.detach().clone())

                    # Clean up
                    del pred_trajectory, ode_state, pred_m, pred_z, target_m, target_z
                    clear_memory()

                except Exception as e:
                    print(f"      ❌ {protein_id}: {str(e)[:50]}...")
                    continue

            if step_comparisons > 0:
                avg_step_loss = step_loss / step_comparisons
                total_loss += avg_step_loss
                num_evaluations += 1

        # Clean up
        del protein_states
        clear_memory()

    ode_func.train()  # Set model back to training mode

    if num_evaluations > 0:
        final_loss = total_loss / num_evaluations
        print(f"    📊 {split.capitalize()} loss: {final_loss:.4f}")
        return final_loss
    else:
        return 0.0


def train_step_batch_efficient():
    """Batch-efficient training step - process all proteins at each time step from TRAINING split only"""
    # FIXED: Only load training data
    dataset = get_dataset(DATA_DIR, split='training')

    if not dataset:
        print("❌ No proteins found in training data directory!")
        return torch.tensor(0.0, device=device)

    # Filter to test protein if specified
    if args.test_protein and args.test_protein in dataset:
        dataset = [args.test_protein]
        print(f"Testing on single protein: {args.test_protein}")

    print(f"  🧬 Training on {len(dataset)} proteins from TRAINING split")

    # Load initial states for all proteins
    print(f"    📥 Loading initial states...")
    protein_states = {}
    successful_proteins = []

    for protein_id in dataset:
        try:
            m_init, z_init = load_initial_input(protein_id)
            protein_states[protein_id] = (m_init, z_init)
            successful_proteins.append(protein_id)
        except Exception as e:
            print(f"      ❌ {protein_id}: Failed to load - {e}")
            continue

    if not successful_proteins:
        print("    ❌ No proteins loaded successfully!")
        return torch.tensor(0.0, device=device)

    print(f"    📊 Successfully loaded {len(successful_proteins)} proteins")

    total_loss = 0
    num_time_steps = 0

    # Process each time step for all proteins together
    for time_step in range(1, args.num_time_points):
        print(f"\n    🎯 Time step {time_step}/{args.num_time_points - 1}")
        print_memory_stats(f"Before time step {time_step}")

        try:
            clear_memory()

            # Create time points for this step
            t_step = torch.tensor([0.0, float(time_step) / (args.num_time_points - 1)], device=device)

            step_loss = torch.tensor(0.0, device=device)
            step_comparisons = 0
            failed_proteins = []

            # Process each protein at this time step WITH MEMORY CLEANUP
            for i, protein_id in enumerate(successful_proteins):
                try:
                    # Clear GPU memory between proteins to prevent accumulation
                    if i > 0:
                        clear_memory()

                    m_current, z_current = protein_states[protein_id]

                    with autocast(enabled=args.use_amp):
                        # Run ODE for just this time step
                        ode_state = (m_current.clone(), z_current.clone())

                        if args.reduced_precision_integration:
                            pred_trajectory = odeint(
                                ode_func, ode_state, t_step, method=args.integrator, rtol=1e-2, atol=1e-2
                            )
                        else:
                            pred_trajectory = odeint(
                                ode_func, ode_state, t_step, method=args.integrator
                            )

                        # Get prediction at final time point (index 1)
                        pred_m = pred_trajectory[0][1]
                        pred_z = pred_trajectory[1][1]

                        # Generate ground truth for this time step
                        target_m, target_z = iter_manager.generate_next_iteration(protein_id, time_step - 1)

                        # Compute loss for this protein at this time step
                        protein_loss, msa_loss, pair_loss = balanced_loss_fn(pred_m, target_m, pred_z, target_z)
                        step_loss += protein_loss
                        step_comparisons += 1

                        # Update protein state for next time step
                        protein_states[protein_id] = (pred_m.detach().clone(), pred_z.detach().clone())

                        print(f"      ✅ {protein_id}: Loss={protein_loss:.3f}")

                        # Clean up tensors immediately
                        del pred_trajectory, ode_state, pred_m, pred_z, target_m, target_z, protein_loss
                        clear_memory()

                except Exception as e:
                    print(f"      ❌ {protein_id}: {str(e)[:50]}...")
                    failed_proteins.append(protein_id)
                    clear_memory()
                    continue

            # Remove failed proteins from further processing
            for protein_id in failed_proteins:
                if protein_id in successful_proteins:
                    successful_proteins.remove(protein_id)
                if protein_id in protein_states:
                    del protein_states[protein_id]

            # Clean up generated files from PREVIOUS time step
            if time_step > 1:
                cleanup_time_step = time_step - 1
                cleanup_count = iter_manager.cleanup_specific_timestep(cleanup_time_step)

            if step_comparisons > 0:
                avg_step_loss = step_loss / step_comparisons
                total_loss += avg_step_loss
                num_time_steps += 1
                print(f"      📊 Average loss: {avg_step_loss:.3f} ({step_comparisons} proteins)")

                print_memory_stats(f"After time step {time_step}")
            else:
                print(f"      ⚠️  No proteins completed time step {time_step}")

        except Exception as e:
            print(f"      ❌ Time step {time_step} failed: {e}")
            continue

    # Final cleanup
    final_cleanup_time_step = args.num_time_points - 1
    if final_cleanup_time_step > 0:
        cleanup_count = iter_manager.cleanup_specific_timestep(final_cleanup_time_step)

    # Clean up protein states
    del protein_states
    clear_memory()

    # Calculate final average loss
    if num_time_steps > 0:
        final_loss = total_loss / num_time_steps
        print(f"    📊 Final training loss: {final_loss:.4f} across {num_time_steps} time steps")
        return final_loss
    else:
        print(f"    ⚠️  No time steps completed successfully")
        return torch.tensor(0.0, device=device)


def train():
    """Main training loop with proper train/validation split"""
    print(f"🚀 Evoformer ODE Training with Validation")
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"💾 Output directory: {OUTPUT_DIR}")
    print(f"🧹 Auto-cleanup enabled: generated files will be deleted after each time step")

    training_history = []
    validation_history = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
        print("=" * 50)

        clear_memory()
        print_memory_stats(f"Start of Epoch {epoch + 1}")

        # TRAINING STEP
        print(f"\n📚 Training Phase")
        optimizer.zero_grad()
        train_loss = train_step_batch_efficient()

        if isinstance(train_loss, torch.Tensor) and train_loss.item() > 0:
            # Backpropagate
            if args.use_amp:
                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_loss = train_loss.item()
        else:
            epoch_train_loss = 0.0

        # VALIDATION STEP
        print(f"\n🔍 Validation Phase")
        epoch_val_loss = evaluate_step(split='validation')

        # Clean up any remaining generated files
        print(f"\n🧹 Performing end-of-epoch cleanup...")
        iter_manager.cleanup_all_generated_files(keep_blocks=[0])

        training_history.append(epoch_train_loss)
        validation_history.append(epoch_val_loss)

        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Training Loss: {epoch_train_loss:.6f}")
        print(f"   Validation Loss: {epoch_val_loss:.6f}")

        # Early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ode_func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'args': vars(args),
            }, best_model_path)
            print(f"   💾 New best model saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{args.patience}")

        print_memory_stats(f"End of Epoch {epoch + 1}")

        # Save regular checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f"evoformer_ode_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': ode_func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'args': vars(args),
            'training_history': training_history,
            'validation_history': validation_history
        }, checkpoint_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early stopping triggered! No improvement for {args.patience} epochs")
            break

        # Force cleanup after each epoch
        del train_loss
        clear_memory()

    # Final cleanup
    print(f"\n🧹 Performing final cleanup...")
    iter_manager.cleanup_all_generated_files(keep_blocks=[0])

    # TESTING PHASE (optional)
    print(f"\n🎯 Final Testing Phase")
    test_loss = evaluate_step(split='testing')

    # Save final summary
    summary = {
        'final_train_loss': training_history[-1] if training_history else 0,
        'final_val_loss': validation_history[-1] if validation_history else 0,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'training_history': training_history,
        'validation_history': validation_history,
        'configuration': vars(args)
    }

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎯 Training Complete!")
    print(f"📈 Training loss: {training_history}")
    print(f"📉 Validation loss: {validation_history}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    print(f"🧪 Final test loss: {test_loss:.6f}")
    print(f"📄 Summary saved: {summary_path}")
    print(f"🧹 All generated iteration files have been cleaned up")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("🧹 Performing cleanup...")
        try:
            iter_manager.cleanup_all_generated_files(keep_blocks=[0])
        except:
            pass
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🧹 Performing cleanup...")
        try:
            iter_manager.cleanup_all_generated_files(keep_blocks=[0])
        except:
            pass
        import traceback

        traceback.print_exc()