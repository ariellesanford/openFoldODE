import os
import torch
from torchdiffeq import odeint_adjoint as odeint  # Use adjoint method for memory efficiency
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc
from torch.cuda.amp import autocast, GradScaler
import gc

# === Configuration ===
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = 16  # Reduced from 64 to save memory
learning_rate = 1e-3
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data"

# Memory management settings
torch.backends.cudnn.benchmark = True
use_amp = True  # Use mixed precision
segment_size = 10  # Process trajectory in segments to save memory


def get_dataset(input_dir):
    """ Get all protein folder names. """
    return [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]


def load_initial_input(protein_id):
    """
    Load the initial M and Z tensors for block 0.
    """
    protein_dir = os.path.join(DATA_DIR, protein_id, "recycle_0")
    m_path = os.path.join(protein_dir, "m_block_0.pt")
    z_path = os.path.join(protein_dir, "z_block_0.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Initial block not found for {protein_id}")

    # Load tensors
    m_init = torch.load(m_path).to(device)  # (s_c, r, c_m)
    z_init = torch.load(z_path).to(device)  # (r, r, c_z)

    return m_init, z_init


def load_block(protein_id, block_index):
    """
    Load M and Z tensors for a specific block index.
    Adjusts dimensions to ensure compatibility.
    """
    protein_dir = os.path.join(DATA_DIR, protein_id, "recycle_0")
    m_path = os.path.join(protein_dir, f"m_block_{block_index}.pt")
    z_path = os.path.join(protein_dir, f"z_block_{block_index}.pt")

    if not os.path.exists(m_path) or not os.path.exists(z_path):
        raise FileNotFoundError(f"Block {block_index} not found for {protein_id}")

    # Load tensors
    m = torch.load(m_path).to(device)  # (1, s_c, r, c_m)
    z = torch.load(z_path).to(device)  # (1, r, r, c_z)

    # Remove batch dimension for consistency
    if m.dim() == 4:
        m = m.squeeze(0)  # (s_c, r, c_m)
    if z.dim() == 4:
        z = z.squeeze(0)  # (r, r, c_z)

    return m, z


def run_evoformer_block(m, z):
    """
    Placeholder for Evoformer block.
    Adjust dimensions as needed to match your implementation.
    """
    m_out = m + torch.randn_like(m) * 0.01
    z_out = z + torch.randn_like(z) * 0.01
    return m_out, z_out


def loss_fn(pred, target):
    """ MSE Loss function. """
    return torch.nn.functional.mse_loss(pred, target)


# === Neural ODE Initialization ===
ode_func = EvoformerODEFunc(c_m, c_z, hidden_dim).to(device)
optimizer = optim.Adam(ode_func.parameters(), lr=learning_rate)
scaler = GradScaler(enabled=use_amp)


def clear_gpu_memory():
    """Clear GPU cache to free up memory"""
    torch.cuda.empty_cache()
    gc.collect()


def train_step(protein_id):
    """
    Single training step for one protein with memory-efficient processing.
    Process the trajectory in segments to reduce memory usage.
    """
    m_init, z_init = load_initial_input(protein_id)
    ode_state = (m_init, z_init)

    # Define full time grid
    full_t_grid = torch.linspace(0, 1, 49).to(device)

    total_loss = 0

    # Process the trajectory in segments to save memory
    for seg_start in range(0, len(full_t_grid) - 1, segment_size):
        seg_end = min(seg_start + segment_size + 1, len(full_t_grid))

        # Create segment time grid
        t_grid_segment = full_t_grid[seg_start:seg_end]

        # If not the first segment, load initial state from previous computation
        if seg_start > 0:
            try:
                m_state, z_state = load_block(protein_id, seg_start)
                ode_state = (m_state, z_state)
            except FileNotFoundError:
                # Continue with previous state if block not found
                pass

        # Use mixed precision and adjoint method for memory-efficient integration
        with autocast(enabled=use_amp):
            pred_trajectory = odeint(
                ode_func,
                ode_state,
                t_grid_segment,
                method='rk4',  # More memory efficient than rk4 # or dopri5
                options={'step_size': 1e-3},
                adjoint_options={'step_size': 1e-3}
            )

        # Process each time point in the segment (except the first which is the initial state)
        seg_loss = 0
        for i in range(1, len(t_grid_segment)):
            global_idx = seg_start + i

            # Skip if it's beyond our available blocks
            if global_idx >= len(full_t_grid):
                continue

            # Load ground truth for the current block
            try:
                gt_m, gt_z = load_block(protein_id, global_idx)
            except FileNotFoundError:
                print(f"Skipping block {global_idx} for {protein_id} (not found)")
                continue

            # Get Neural ODE predictions at current time point
            pred_m = pred_trajectory[0][i]
            pred_z = pred_trajectory[1][i]

            # Calculate loss for this time point
            with autocast(enabled=use_amp):
                loss = loss_fn(pred_m, gt_m) + loss_fn(pred_z, gt_z)
                seg_loss += loss

        # Add segment loss to total
        total_loss += seg_loss

        # Free up memory
        del pred_trajectory, seg_loss
        clear_gpu_memory()

    return total_loss


def train(input_dir):
    """
    Main training loop with memory optimizations.
    """
    dataset = get_dataset(input_dir)
    print(f"Training on {len(dataset)} proteins from {input_dir}")

    for epoch in range(epochs):
        epoch_loss = 0

        for idx, protein_id in enumerate(dataset):
            print(f"Processing protein {idx + 1}/{len(dataset)}: {protein_id}")

            # Clear memory before processing each protein
            clear_gpu_memory()

            optimizer.zero_grad()

            # Use mixed precision training
            with autocast(enabled=use_amp):
                loss = train_step(protein_id)

            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Force release memory
            del loss
            clear_gpu_memory()

            print(f"  - Current loss: {epoch_loss / (idx + 1):.6f}")

        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss / len(dataset):.6f}")

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': ode_func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pt')

    print("Training complete!")


# === Run Training ===
if __name__ == "__main__":
    # Set memory allocation settings to minimize fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Try to limit GPU memory usage
    if torch.cuda.is_available():
        # Get total GPU memory
        total_mem = torch.cuda.get_device_properties(0).total_memory
        # Reserve only 80% of available memory
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"Using device: {device} with {total_mem / 1e9:.2f} GB total memory")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        train(DATA_DIR)
    except Exception as e:
        print(f"Error during training: {e}")
        # Print GPU memory stats for debugging
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")