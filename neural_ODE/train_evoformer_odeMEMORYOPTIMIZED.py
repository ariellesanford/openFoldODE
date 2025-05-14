import os
import gc
import torch
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# === Memory Optimization Configuration ===
# Set memory split size to avoid fragmentation
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Increased from 32
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")


# Define a memory tracking function
def print_memory_stats(label=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(f"=== Memory Stats {label} ===")
    print(f"Allocated Memory: {allocated:.2f} MiB")
    print(f"Reserved Memory: {reserved:.2f} MiB")
    print(f"Max Memory Allocated: {max_allocated:.2f} MiB")
    print(f"Max Memory Reserved: {max_reserved:.2f} MiB")
    print("=" * 30)


# Trigger CUDA initialization by creating a tensor on the GPU
_ = torch.tensor([0.0], device="cuda")
print_memory_stats("After CUDA init")


# Force garbage collection and clear CUDA cache
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


clear_memory()

# === Configuration ===
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = 8  # Reduced from 16 to save memory
learning_rate = 1e-3
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data"
USE_AMP = True  # Use mixed precision
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over multiple steps
CHUNK_SIZE = 10  # Process time steps in smaller chunks to save memory

# Setup gradient scaler for mixed precision
scaler = GradScaler(enabled=USE_AMP)


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

    # Reduce cluster size more aggressively
    max_clusters = 64  # Reduced from 128 to save memory
    m_init = m_init[:max_clusters]

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
    m = m.squeeze(0)  # (s_c, r, c_m)
    z = z.squeeze(0)  # (r, r, c_z)

    # Reduce cluster size
    max_clusters = 64  # Reduced from 128 to save memory
    m = m[:max_clusters]
    return m, z


def loss_fn(pred, target):
    """ MSE Loss function. """
    return torch.nn.functional.mse_loss(pred, target)


# === Neural ODE with Memory Optimizations ===
# Create a more memory-efficient ODE function with checkpointing
class CheckpointedEvoformerODEFunc(torch.nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, t, state):
        # Use checkpointing to save memory during backprop
        return checkpoint(self.ode_func, t, state)


# Initialize model with reduced parameters
ode_func = EvoformerODEFunc(c_m, c_z, hidden_dim).to(device)
checkpointed_ode_func = CheckpointedEvoformerODEFunc(ode_func)
optimizer = optim.Adam(ode_func.parameters(), lr=learning_rate)

# Use fewer integration points
num_points = 25  # Reduced from 49
t_grid_full = torch.linspace(0, 1, num_points).to(device)


def chunk_integration(func, y0, t, chunk_size=CHUNK_SIZE):
    """
    Perform integration in chunks to reduce memory usage.
    """
    chunks = []
    for i in range(0, len(t), chunk_size):
        # Get current chunk of time points
        t_chunk = t[i:i + chunk_size]
        if i == 0:
            # For the first chunk, use the initial state
            y_chunk = y0
        else:
            # For subsequent chunks, use the final state from the previous chunk
            y_chunk = chunks[-1][-1]

        # Integrate this chunk
        with torch.no_grad():  # No need to track gradients during forward pass
            chunk_traj = odeint(func, y_chunk, t_chunk, method='dopri5', rtol=1e-2, atol=1e-3)
        chunks.append(chunk_traj)

    # Concatenate the chunks excluding duplicate states
    result = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            result.append(chunk)
        else:
            # Skip the first state in subsequent chunks (it's the same as the last state of the previous chunk)
            result.append(chunk[:, 1:])

    return torch.cat(result, dim=1)


def train_step(protein_id, step_idx):
    """
    Single training step for one protein with memory optimizations.
    """
    clear_memory()

    m_init, z_init = load_initial_input(protein_id)
    ode_state = (m_init, z_init)

    # Process fewer time steps per batch to save memory
    batch_size = 5  # Process 5 time steps at a time
    total_loss = 0

    # Calculate loss for chunks of time steps
    for batch_start in range(1, len(t_grid_full), batch_size):
        batch_end = min(batch_start + batch_size, len(t_grid_full))
        t_grid_batch = t_grid_full[batch_start - 1:batch_end]

        # Clear gradients
        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            # Integrate Neural ODE for this batch
            # pred_trajectory = odeint(
            #     ode_func,
            #     ode_state,
            #     t_grid_batch,
            #     method='dopri5',  # Use dopri5 instead of rk4 for memory efficiency
            #     rtol=1e-2,  # Relaxed tolerances to reduce computation
            #     atol=1e-3
            # )
            pred_trajectory = odeint(
                ode_func,
                ode_state,
                t_grid_batch,
                method='rk4',  # More memory efficient than rk4 # or dopri5
                options={'step_size': 1e-3}
            )

            # Compute loss for each time step in the batch
            batch_loss = 0
            for i in range(1, len(t_grid_batch)):
                time_idx = batch_start - 1 + i
                try:
                    gt_m, gt_z = load_block(protein_id, time_idx)
                except FileNotFoundError:
                    print(f"Skipping block {time_idx} for {protein_id} (not found)")
                    continue

                # Get predictions
                pred_m, pred_z = pred_trajectory[0][i], pred_trajectory[1][i]

                # Compute loss
                loss = loss_fn(pred_m, gt_m) + loss_fn(pred_z, gt_z)
                batch_loss += loss

            if batch_loss > 0:
                batch_loss = batch_loss / (batch_end - batch_start)
                total_loss += batch_loss.item()

        # Scale gradients and backpropagate
        if USE_AMP:
            scaler.scale(batch_loss).backward()
            if (step_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            batch_loss.backward()
            if (step_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()

        # Free memory
        del pred_trajectory, batch_loss
        clear_memory()

    # Free remaining memory
    del m_init, z_init, ode_state
    clear_memory()

    return total_loss


def train(input_dir):
    """
    Main training loop with memory optimizations.
    """
    dataset = get_dataset(input_dir)
    print(f"Training on {len(dataset)} proteins from {input_dir}")

    for epoch in range(epochs):
        clear_memory()
        print_memory_stats(f"Start of Epoch {epoch + 1}")

        epoch_loss = 0
        for step_idx, protein_id in enumerate(dataset):
            try:
                print(f"Processing protein {protein_id} - Step {step_idx + 1}/{len(dataset)}")
                loss = train_step(protein_id, step_idx)
                epoch_loss += loss
                print(f"  - Loss: {loss}")
                print_memory_stats(f"After protein {protein_id}")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM for protein {protein_id}. Skipping...")
                    clear_memory()
                    continue
                else:
                    raise e

        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss / len(dataset)}")
        # Save checkpoint
        checkpoint_path = f"evoformer_ode_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': ode_func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


# === Run Training ===
if __name__ == "__main__":
    try:
        train(DATA_DIR)
    except Exception as e:
        print(f"Error during training: {e}")
        # Print final memory stats
        print_memory_stats("Error state")