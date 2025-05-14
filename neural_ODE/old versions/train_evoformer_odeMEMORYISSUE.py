import os

# Set memory split size to avoid fragmentation
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" #"max_split_size_mb:32"
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")

import torch

# Trigger CUDA initialization by creating a tensor on the GPU
_ = torch.tensor([0.0], device="cuda")
# Print memory stats
print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MiB")
print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB")
print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MiB")


from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc
from torch.cuda.amp import autocast, GradScaler
import gc


torch.cuda.empty_cache() #memory
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()


# === Configuration ===
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = 16 #64  #memory
learning_rate = 1e-3
epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data"

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

    # Reduce cluster size
    max_clusters = 128  # Reduce to 128 clusters  #memory
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
    max_clusters = 128  # Reduce to 128 clusters  #memory
    m = m[:max_clusters]
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
t_grid = torch.linspace(0, 1, 49).to(device)  # 49 time points for 48 layers

def train_step(protein_id):
    """
    Single training step for one protein.
    """
    m_init, z_init = load_initial_input(protein_id)
    ode_state = (m_init, z_init)
    gt_m, gt_z = m_init.clone(), z_init.clone()

    with autocast():   #memory
        torch.cuda.empty_cache() #memory
        gc.collect()

        # Integrate Neural ODE
        pred_trajectory = odeint(ode_func, ode_state, t_grid, method='rk4')

        total_loss = 0
        for i in range(1, len(t_grid)):
            # Load ground truth for the current block
            try:
                gt_m, gt_z = load_block(protein_id, i)  # Temporary until evoformer_iter is called
            except FileNotFoundError:
                print(f"Skipping block {i} for {protein_id} (not found)")
                continue

            # Get Neural ODE predictions at the current time point
            pred_m = pred_trajectory[0][i]  # Shape: (516, 28, 256)
            pred_z = pred_trajectory[1][i]  # Shape: (28, 28, 128)

            # Compute loss
            loss = loss_fn(pred_m, gt_m) + loss_fn(pred_z, gt_z)
            total_loss += loss

    del m_init, z_init, ode_state, gt_m, gt_z, pred_trajectory, loss, pred_m, pred_z
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss

def train(input_dir):
    """
    Main training loop.
    """
    dataset = get_dataset(input_dir)
    print(f"Training on {len(dataset)} proteins from {input_dir}")

    for epoch in range(epochs):
        # Clear memory at the start of each epoch
        torch.cuda.empty_cache()   #memory
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

        epoch_loss = 0
        for protein_id in dataset:
            optimizer.zero_grad()
            loss = train_step(protein_id)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss / len(dataset)}")

# === Run Training ===
if __name__ == "__main__":
    train(DATA_DIR)
