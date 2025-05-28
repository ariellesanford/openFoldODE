import os
import gc
import torch
import argparse
import sys
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc  # Import from existing module
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

# Function to get project root directory
def get_project_root():
    """Get the path to the project root directory."""
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The current file should be in the neural_ODE directory, so current_dir is the project root
    return current_dir


# === Parse command line arguments for memory optimizations ===
parser = argparse.ArgumentParser(description='Train Evoformer ODE with configurable memory optimizations')

# Memory optimization options
parser.add_argument('--memory_split_size', type=int, default=128,
                    help='Memory split size (MB) to avoid fragmentation')
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Use Automatic Mixed Precision')
parser.add_argument('--no-use_amp', dest='use_amp', action='store_false',
                    help='Disable Automatic Mixed Precision')
parser.add_argument('--use_checkpoint', action='store_true', default=False,
                    help='Use gradient checkpointing')
parser.add_argument('--no-use_checkpoint', dest='use_checkpoint', action='store_false',
                    help='Disable gradient checkpointing')
parser.add_argument('--gradient_accumulation', type=int, default=4,
                    help='Number of steps to accumulate gradients over (1 disables)')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='Size of chunks for time integration (0 disables chunking)')
parser.add_argument('--reduced_precision_integration', action='store_true', default=False,
                    help='Use reduced precision for ODE integration')
parser.add_argument('--no-reduced_precision_integration', dest='reduced_precision_integration',
                    action='store_false', help='Disable reduced precision for ODE integration')
parser.add_argument('--clean_memory', action='store_true', default=False,
                    help='Aggressively clean memory')
parser.add_argument('--no-clean_memory', dest='clean_memory', action='store_false',
                    help='Disable aggressive memory cleaning')
parser.add_argument('--reduced_cluster_size', type=int, default=64,
                    help='Maximum cluster size (original is 128)')
parser.add_argument('--reduced_hidden_dim', type=int, default=128,
                    help='Hidden dimension size (default is 128 from EvoformerODEFunc)')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate for the optimizer')
parser.add_argument('--num_time_points', type=int, default=25,
                    help='Number of integration time points (original is 49)')
parser.add_argument('--integrator', choices=['dopri5', 'rk4', 'euler'], default='dopri5',
                    help='ODE integrator method')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Batch size for time steps')
parser.add_argument('--use_fast_ode', action='store_true', default=False,
                    help='Use the faster implementation of EvoformerODEFunc')
parser.add_argument('--no-use_fast_ode', dest='use_fast_ode', action='store_false',
                    help='Use the standard implementation of EvoformerODEFunc')
parser.add_argument('--monitor_memory', action='store_true', default=False,
                    help='Print memory usage statistics')
parser.add_argument('--no-monitor_memory', dest='monitor_memory', action='store_false',
                    help='Disable memory usage statistics')


parser.add_argument('--cpu-only', action='store_true', default=False,
                    help='Force CPU-only mode regardless of CUDA availability')
parser.add_argument('--no-cpu-only', dest='cpu_only', action='store_false',
                    help='Disable CPU-only mode (use CUDA if available)')

# Data and output directory options
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to data directory')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Path to output directory')

# Test mode option
parser.add_argument('--test-configs', action='store_true', default=False,
                    help='Run configuration testing instead of training')
parser.add_argument('--test-single-step', action='store_true', default=False,
                    help='Run only one training step for testing')
parser.add_argument('--test-protein', type=str, default=None,
                    help='Specific protein ID to test (use "all" to test all proteins)')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of training epochs')

# Parse arguments
args = parser.parse_args()

# === Memory Optimization Configuration ===
# Set memory split size to avoid fragmentation
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.memory_split_size}"
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")

# Get project root directory
PROJECT_ROOT = get_project_root()

# Check if CPU-only mode is forced
if args.cpu_only:
    print("Forcing CPU-only mode (CUDA disabled)")
    device = "cpu"
    # Disable features that only work with CUDA
    args.use_amp = False
else:
    # Set device based on CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, using CPU")
        args.use_amp = False


# Define a memory tracking function
def print_memory_stats(label=""):
    if not args.monitor_memory:
        return

    if device == "cpu":
        print(f"=== Memory Stats {label} (CPU mode) ===")
        print("Memory monitoring is limited in CPU mode")
        import psutil
        process = psutil.Process(os.getpid())
        print(f"Current Process Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MiB")
        print("=" * 30)
        return

    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2

        try:
            max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
            max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
        except RuntimeError:
            # Handle case where peak stats aren't available
            max_allocated = allocated
            max_reserved = reserved

        print(f"=== Memory Stats {label} ===")
        print(f"Allocated Memory: {allocated:.2f} MiB")
        print(f"Reserved Memory: {reserved:.2f} MiB")
        print(f"Max Memory Allocated: {max_allocated:.2f} MiB")
        print(f"Max Memory Reserved: {max_reserved:.2f} MiB")
        print("=" * 30)
    except Exception as e:
        print(f"=== Memory Stats {label} (Error) ===")
        print(f"Could not get memory stats: {e}")
        print("=" * 30)


# Trigger CUDA initialization if using CUDA
if device == "cuda":
    try:
        _ = torch.tensor([0.0], device=device)
        print("CUDA initialized successfully")
    except RuntimeError as e:
        print(f"Error initializing CUDA: {e}")
        print("Falling back to CPU mode")
        device = "cpu"
        args.use_amp = False
else:
    print("Using CPU - CUDA initialization skipped")


# Force garbage collection and clear CUDA cache
def clear_memory():
    if not args.clean_memory:
        return

    gc.collect()

    if device == "cuda":
        try:
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError as e:
                print(f"Warning: Could not reset CUDA peak memory stats: {e}")
        except Exception as e:
            print(f"Warning: CUDA error when clearing memory: {e}")


clear_memory()

# === Configuration ===
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = args.reduced_hidden_dim  # Configurable hidden dimension
learning_rate = args.learning_rate

# Use command line data directory if provided, otherwise use default
if args.data_dir is None:
    print("Warning: No data directory specified. Using default.")
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
else:
    DATA_DIR = args.data_dir
    print(f"Using data directory: {DATA_DIR}")

if args.output_dir is None:
    print("Warning: No output directory specified. Using current directory.")
    OUTPUT_DIR = PROJECT_ROOT
else:
    OUTPUT_DIR = args.output_dir
    print(f"Using output directory: {OUTPUT_DIR}")

# Use configured memory optimization settings
USE_AMP = args.use_amp and device == "cuda"  # AMP only works with CUDA
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
CHUNK_SIZE = args.chunk_size

# Setup gradient scaler for mixed precision
scaler = GradScaler(enabled=USE_AMP)

# Print active optimizations
print("\n=== ACTIVE CONFIGURATION ===")
print(f"Device: {device.upper()}")
print(f"Memory Split Size: {args.memory_split_size} MB")
print(f"Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}")
print(f"Gradient Checkpointing: {'Enabled' if args.use_checkpoint else 'Disabled'}")
print(f"Gradient Accumulation: {args.gradient_accumulation} steps")
print(f"Integration Chunking: {'Enabled' if args.chunk_size > 0 else 'Disabled'}")
print(f"Reduced Precision Integration: {'Enabled' if args.reduced_precision_integration else 'Disabled'}")
print(f"Aggressive Memory Cleaning: {'Enabled' if args.clean_memory else 'Disabled'}")
print(f"Memory Usage Monitoring: {'Enabled' if args.monitor_memory else 'Disabled'}")
print(f"Reduced Cluster Size: {args.reduced_cluster_size} (original: 128)")
print(f"Reduced Hidden Dimension: {args.reduced_hidden_dim} (default: 128)")
print(f"Learning Rate: {learning_rate}")
print(f"Number of Time Points: {args.num_time_points} (original: 49)")
print(f"ODE Integrator: {args.integrator}")
print(f"Time Step Batch Size: {args.batch_size}")
print("=" * 30)


def get_dataset(input_dir):
    """ Get all protein folder names that contain evoformer blocks data. """
    datasets = []
    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        # Only include directories that end with '_evoformer_blocks' and contain recycle_0 subdirectory
        if (os.path.isdir(full_path) and
            name.endswith('_evoformer_blocks') and
            os.path.isdir(os.path.join(full_path, 'recycle_0'))):
            # Extract the protein ID (remove '_evoformer_blocks' suffix)
            protein_id = name.replace('_evoformer_blocks', '')
            datasets.append(protein_id)
    return datasets


def load_initial_input(protein_id):
    """
    Load the initial M and Z tensors for block 0.
    """
    # Updated to match the actual directory structure
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, "m_block_0.pt")
    z_path = os.path.join(protein_dir, "z_block_0.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Initial block not found for {protein_id}. Expected paths:\n  {m_path}\n  {z_path}")

    # Load tensors
    m_init = torch.load(m_path, map_location=device)  # (s_c, r, c_m)
    z_init = torch.load(z_path, map_location=device)  # (r, r, c_z)

    # Ensure proper dimensions
    if m_init.dim() == 4 and m_init.size(0) == 1:
        m_init = m_init.squeeze(0)  # Remove batch dimension if present
    if z_init.dim() == 4 and z_init.size(0) == 1:
        z_init = z_init.squeeze(0)  # Remove batch dimension if present

    # Reduce cluster size if enabled
    max_clusters = args.reduced_cluster_size
    m_init = m_init[:max_clusters]
    return m_init, z_init


def load_block(protein_id, block_index):
    """
    Load M and Z tensors for a specific block index.
    Adjusts dimensions to ensure compatibility.
    """
    # Updated to match the actual directory structure
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_evoformer_blocks", "recycle_0")
    m_path = os.path.join(protein_dir, f"m_block_{block_index}.pt")
    z_path = os.path.join(protein_dir, f"z_block_{block_index}.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Block {block_index} not found for {protein_id}. Expected paths:\n  {m_path}\n  {z_path}")

    # Load tensors
    m = torch.load(m_path, map_location=device)  # (1, s_c, r, c_m)
    z = torch.load(z_path, map_location=device)  # (1, r, r, c_z)

    # Remove batch dimension for consistency
    m = m.squeeze(0)  # (s_c, r, c_m)
    z = z.squeeze(0)  # (r, r, c_z)

    # Reduce cluster size if enabled
    max_clusters = args.reduced_cluster_size
    m = m[:max_clusters]
    return m, z


def balanced_loss_fn(pred_m, target_m, pred_z, target_z, msa_weight=1.0, pair_weight=0.1):
    """
    Balanced loss function that accounts for different data scales
    """
    msa_loss = F.mse_loss(pred_m, target_m)
    pair_loss = F.mse_loss(pred_z, target_z)

    # Weight the losses to balance their contributions
    weighted_loss = msa_weight * msa_loss + pair_weight * pair_loss

    return weighted_loss, msa_loss.item(), pair_loss.item()


# === Neural ODE with Configurable Memory Optimizations ===
# Create a ODE function with optional checkpointing
class CheckpointedEvoformerODEFunc(torch.nn.Module):
    def __init__(self, ode_func, use_checkpoint=True):
        super().__init__()
        self.ode_func = ode_func
        self.use_checkpoint = use_checkpoint

    def forward(self, t, state):
        if self.use_checkpoint:
            # Use checkpointing to save memory during backprop
            # Fix: use use_reentrant=False as recommended
            return checkpoint(self.ode_func.forward, t, state, use_reentrant=False)
        else:
            # Direct computation without checkpointing
            return self.ode_func(t, state)


# Initialize model with configurable parameters
if args.use_fast_ode:
    print("Using FAST EvoformerODEFunc implementation")
    from evoformer_ode import EvoformerODEFuncFast as OdeFunction
else:
    print("Using standard EvoformerODEFunc implementation")
    from evoformer_ode import EvoformerODEFunc as OdeFunction

ode_func = OdeFunction(c_m, c_z, hidden_dim).to(device)
checkpointed_ode_func = CheckpointedEvoformerODEFunc(ode_func, use_checkpoint=args.use_checkpoint)
optimizer = optim.Adam(ode_func.parameters(), lr=learning_rate)

# Configure integration time points
num_points = args.num_time_points
t_grid_full = torch.linspace(0, 1, num_points).to(device)


def chunk_integration(func, y0, t, chunk_size=CHUNK_SIZE):
    """
    Perform integration in chunks to reduce memory usage.
    Only used if chunk_size > 0.
    """
    if chunk_size <= 0:
        # Do regular integration if chunking is disabled
        return odeint(func, y0, t, method=args.integrator,
                      rtol=1e-3 if args.reduced_precision_integration else 1e-4,
                      atol=1e-4 if args.reduced_precision_integration else 1e-5)

    chunks = []
    for i in range(0, len(t), chunk_size):
        # Get current chunk of time points
        t_chunk = t[i:i + chunk_size]
        if len(t_chunk) == 1 and i > 0:
            # Skip single point chunks except the first
            continue

        if i == 0:
            # For the first chunk, use the initial state
            y_chunk = y0
        else:
            # For subsequent chunks, use the final state from the previous chunk
            if chunks:
                last_chunk = chunks[-1]
                if isinstance(last_chunk, tuple):
                    # If the result is a tuple of tensors
                    y_chunk = (last_chunk[0][-1], last_chunk[1][-1])
                else:
                    # If it's a single tensor
                    y_chunk = last_chunk[-1]
            else:
                y_chunk = y0

        # Integrate this chunk
        chunk_traj = odeint(
            func,
            y_chunk,
            t_chunk,
            method=args.integrator,
            rtol=1e-3 if args.reduced_precision_integration else 1e-4,
            atol=1e-4 if args.reduced_precision_integration else 1e-5
        )
        chunks.append(chunk_traj)

    # Concatenate the chunks excluding duplicate states
    # Since we have a tuple of (m, z) trajectories, we need to handle them separately
    m_results = []
    z_results = []

    for i, chunk in enumerate(chunks):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            m_chunk, z_chunk = chunk
        else:
            # Handle the case where odeint returns trajectories differently
            m_chunk = chunk[0]
            z_chunk = chunk[1]

        if i == 0:
            m_results.append(m_chunk)
            z_results.append(z_chunk)
        else:
            # Skip the first state in subsequent chunks
            m_results.append(m_chunk[1:])
            z_results.append(z_chunk[1:])

    # Concatenate results
    m_trajectory = torch.cat(m_results, dim=0)
    z_trajectory = torch.cat(z_results, dim=0)

    return (m_trajectory, z_trajectory)


def train_step(protein_id, step_idx):
    """
    Single training step for one protein with configurable memory optimizations.
    NOTE: For neural ODEs, we must process time sequentially. The 'batches' here
    are contiguous chunks of the time sequence, not independent batches.
    """
    clear_memory()

    m_init, z_init = load_initial_input(protein_id)

    ode_state = (m_init, z_init)

    # Configure batch size for time steps
    batch_size = args.batch_size
    total_loss = 0

    # Choose which ODE function to use
    active_ode_func = checkpointed_ode_func if args.use_checkpoint else ode_func

    # Create time batches - these are CONTIGUOUS chunks, not independent batches
    all_time_batches = [(batch_start, min(batch_start + batch_size, len(t_grid_full)))
                        for batch_start in range(1, len(t_grid_full), batch_size)]

    # If test single step, only process first batch of time steps for quick testing
    time_batches = all_time_batches[:1] if args.test_single_step else all_time_batches

    # Calculate loss for contiguous chunks of time steps
    current_state = ode_state  # Track the current state through time

    for batch_idx, (batch_start, batch_end) in enumerate(time_batches):
        # Include the last time point from previous batch for continuity
        start_idx = batch_start - 1 if batch_idx == 0 else batch_start - 1
        t_grid_batch = t_grid_full[start_idx:batch_end]

        # Clear gradients
        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            # Integrate Neural ODE from current state through this time chunk
            if args.chunk_size > 0:
                pred_trajectory = chunk_integration(
                    active_ode_func,
                    current_state,  # Use current state, not initial state
                    t_grid_batch,
                    chunk_size=args.chunk_size
                )
            else:
                pred_trajectory = odeint(
                    active_ode_func,
                    current_state,  # Use current state, not initial state
                    t_grid_batch,
                    method='euler',  # CHANGED: use stable integration
                    rtol=1e-2,  # CHANGED: relaxed tolerances
                    atol=1e-2
                )

            # Update current state to the final state of this trajectory
            # Detach to prevent gradients from flowing back through previous chunks
            if isinstance(pred_trajectory, tuple):
                current_state = (pred_trajectory[0][-1].detach(), pred_trajectory[1][-1].detach())
            else:
                # Handle different output formats from different integrators
                current_state = (pred_trajectory[0][-1].detach(), pred_trajectory[1][-1].detach())

            # Compute loss for each time step in the batch
            batch_loss = 0
            # Skip first time point if it's not the first batch (it's from previous batch)
            start_i = 1 if batch_idx == 0 else 0

            for i in range(start_i, len(t_grid_batch)):
                time_idx = start_idx + i
                try:
                    gt_m, gt_z = load_block(protein_id, time_idx)
                except FileNotFoundError:
                    print(f"Skipping block {time_idx} for {protein_id} (not found)")
                    continue

                # Get predictions - Make sure i is within bounds
                if isinstance(pred_trajectory, tuple):
                    # Make sure i is within bounds of pred_trajectory
                    if i < pred_trajectory[0].shape[0] and i < pred_trajectory[1].shape[0]:
                        pred_m, pred_z = pred_trajectory[0][i], pred_trajectory[1][i]
                    else:
                        print(f"Warning: Index {i} out of bounds for pred_trajectory with shape "
                              f"{pred_trajectory[0].shape[0]}, {pred_trajectory[1].shape[0]}. Skipping...")
                        continue
                else:
                    # Handle different output formats for different integrators
                    if i < pred_trajectory[0].shape[0] and i < pred_trajectory[1].shape[0]:
                        pred_m, pred_z = pred_trajectory[0][i], pred_trajectory[1][i]
                    else:
                        print(f"Warning: Index {i} out of bounds for pred_trajectory with shape "
                              f"{pred_trajectory[0].shape[0]}, {pred_trajectory[1].shape[0]}. Skipping...")
                        continue

                # FIXED: Compute balanced loss
                step_loss, msa_loss, pair_loss = balanced_loss_fn(pred_m, gt_m, pred_z, gt_z)
                batch_loss += step_loss  # FIXED: use step_loss instead of undefined 'loss'

                # # Optional: Print component losses for monitoring
                # if i % 3 == 0:  # Every 3rd time step
                #     print(f"    Time {time_idx}: Total={step_loss:.1f}, MSA={msa_loss:.1f}, Pair={pair_loss:.1f}")

            # Adjust loss based on batch size for consistency
            if batch_loss > 0:
                batch_loss = batch_loss / (len(t_grid_batch) - start_i)
                total_loss += batch_loss.item()

        # FIXED: Scale gradients and backpropagate with gradient clipping
        if USE_AMP:
            scaler.scale(batch_loss).backward()
            # ADD gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            if args.gradient_accumulation <= 1 or (step_idx + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            batch_loss.backward()
            # ADD gradient clipping
            torch.nn.utils.clip_grad_norm_(ode_func.parameters(), max_norm=1.0)
            if args.gradient_accumulation <= 1 or (step_idx + 1) % args.gradient_accumulation == 0:
                optimizer.step()

        # Free memory
        del pred_trajectory, batch_loss
        clear_memory()

    # Free remaining memory
    del m_init, z_init, ode_state, current_state
    clear_memory()

    return total_loss


def train(input_dir):
    """
    Main training loop with configurable memory optimizations.
    """
    dataset = get_dataset(input_dir)
    print(f"Training on {len(dataset)} proteins from {input_dir}")

    # If test single step mode is active
    if args.test_single_step:
        epochs_to_run = 1
        if args.test_protein:
            if args.test_protein.lower() == "all":
                # Test on all proteins in the dataset
                dataset_to_process = dataset
                print(f"Testing all {len(dataset)} proteins in dataset")
            elif args.test_protein in dataset:
                dataset_to_process = [args.test_protein]
                print(f"Testing single protein: {args.test_protein}")
            else:
                print(f"Error: Protein {args.test_protein} not found in dataset")
                print(f"Available proteins: {dataset}")
                return
        else:
            dataset_to_process = dataset[:1]
            print(f"Testing first protein: {dataset_to_process[0]}")
    else:
        epochs_to_run = args.epochs  # Use configurable epochs parameter
        dataset_to_process = dataset

    for epoch in range(epochs_to_run):
        clear_memory()

        epoch_loss = 0
        max_mem_allocated = 0
        max_mem_reserved = 0

        for step_idx, protein_id in enumerate(dataset_to_process):
            try:
                print(f"Processing protein {protein_id} - Step {step_idx + 1}/{len(dataset_to_process)}")
                loss = train_step(protein_id, step_idx)
                epoch_loss += loss
                print(f"  - Loss: {loss}")

                # Track maximum memory usage across all proteins
                if device == "cuda":
                    curr_mem_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
                    curr_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
                    max_mem_allocated = max(max_mem_allocated, curr_mem_allocated)
                    max_mem_reserved = max(max_mem_reserved, curr_mem_reserved)

                #print_memory_stats(f"After protein {protein_id}")

            except RuntimeError as e:
                if device == "cuda" and "CUDA out of memory" in str(e):
                    print(f"CUDA OOM for protein {protein_id}. Skipping...")
                    # Print memory usage when OOM occurs
                    print_memory_stats("At OOM error")
                    clear_memory()
                    continue
                else:
                    raise e

        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss / len(dataset_to_process)}")

        # Print maximum memory usage across all proteins
        if len(dataset_to_process) > 1:
            print(f"Maximum Memory Usage Across All Proteins:")
            if device == "cuda":
                print(f"  Max Memory Allocated: {max_mem_allocated:.2f} MiB")
                print(f"  Max Memory Reserved: {max_mem_reserved:.2f} MiB")
            else:
                print("  Memory tracking not available in CPU mode")

        # Only save checkpoints in full training mode
        if not args.test_single_step:
            # Save checkpoint
            checkpoint_path = os.path.join(OUTPUT_DIR, f"evoformer_ode_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ode_func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


# === Main Entry Point ===
if __name__ == "__main__":
    try:
        # Enable memory profiling if available
        try:
            from pytorch_memlab import MemReporter

            reporter = None

            if device == "cuda" and args.monitor_memory:
                reporter = MemReporter()
                # Profile initial memory state silently
                reporter.report()
        except ImportError:
            reporter = None
            if args.monitor_memory:
                print("pytorch_memlab not available, detailed memory monitoring disabled")

        # Choose whether to train or test configurations based on the argument
        if args.test_configs:
            print("\n=== Configuration testing is handled by memory_config_tester.py ===")
            print("Please run memory_config_tester.py directly for configuration testing.")
            sys.exit(0)
        else:
            # Run normal training
            train(DATA_DIR)

        # Profile final memory state
        if reporter is not None and args.monitor_memory:
            print("\n=== FINAL MEMORY PROFILE ===")
            reporter.report()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        # Print final memory stats
        print_memory_stats("Error state")