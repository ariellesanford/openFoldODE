import os
import torch
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc

# === Configuration ===
c_m = 256  # MSA (M) channels
c_z = 128  # Pair (Z) channels
hidden_dim = 64
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

    # Integrate Neural ODE
    pred_trajectory = odeint(ode_func, ode_state, t_grid, method='rk4')

    total_loss = 0

    for i in range(1, len(t_grid)):
        # Load ground truth for the current block
        try:
            gt_m, gt_z = load_block(protein_id, i)  #temporary until i call evoformer_iter
        except FileNotFoundError:
            print(f"Skipping block {i} for {protein_id} (not found)")
            continue

        # Get Neural ODE prediction at current time point
        pred_m, pred_z = pred_trajectory[i]

        # Compute loss
        loss = loss_fn(pred_m, gt_m) + loss_fn(pred_z, gt_z)
        total_loss += loss

    return total_loss

def train(input_dir):
    """
    Main training loop.
    """
    dataset = get_dataset(input_dir)
    print(f"Training on {len(dataset)} proteins from {input_dir}")

    for epoch in range(epochs):
        epoch_loss = 0
        for protein_id in dataset:
            optimizer.zero_grad()
            try:
                loss = train_step(protein_id)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")

        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss / len(dataset)}")

# === Run Training ===
if __name__ == "__main__":
    train(DATA_DIR)
