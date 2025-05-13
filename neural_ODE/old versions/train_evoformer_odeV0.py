import torch
from torchdiffeq import odeint
import torch.optim as optim
from evoformer_ode import EvoformerODEFunc
import os
# === Configuration ===
msa_dim = 256         # Keep as is (based on data format)
pair_dim = 128        # Keep as is (based on data format)
hidden_dim = 64       # Reduced to 64 to speed up training and reduce memory usage
learning_rate = 1e-3  # Increased to speed up convergence
epochs = 5            # Reduced to 5 for a quick test
num_layers = 48
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Directory Paths ===
DATA_DIR = "/home/visitor/PycharmProjects/openFold/neural_ODE/data"

# === Initialize the Neural ODE ===
ode_func = EvoformerODEFunc(msa_dim, pair_dim, hidden_dim).to(device)
optimizer = optim.Adam(ode_func.parameters(), lr=learning_rate)

# === Define Time Grid ===
t_grid = torch.linspace(0, 1, num_layers + 1).to(device)  # 49 points for 48 layers


def get_dataset(input_dir):
    """ Return all folder names in the input directory as dataset IDs. """
    return [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]


def load_initial_input(protein_id):
    """
    Load the initial MSA and pair tensors for the given protein ID.

    Args:
        protein_id (str): The folder name corresponding to the protein (e.g., "4cue_A").

    Returns:
        tuple: (msa_init, pair_init) tensors
    """
    protein_dir = os.path.join(DATA_DIR, f"{protein_id}_A_evoformer_blocks")

    # Construct paths
    msa_path = os.path.join(protein_dir, "m_block_0.pt")
    pair_path = os.path.join(protein_dir, "z_block_0.pt")

    # Ensure files exist
    if not os.path.exists(msa_path):
        raise FileNotFoundError(f"MSA file not found: {msa_path}")
    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"Pair file not found: {pair_path}")

    # Load tensors
    msa_init = torch.load(msa_path).to(device)
    pair_init = torch.load(pair_path).to(device)

    return msa_init, pair_init


def run_evoformer_block(msa, pair):
    # Replace with your actual Evoformer block call
    msa_out = msa + torch.randn_like(msa) * 0.01
    pair_out = pair + torch.randn_like(pair) * 0.01
    return msa_out, pair_out

def loss_fn(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

def train_step(protein_id):
    # Load initial input for this sequence
    msa_init, pair_init = load_initial_input(protein_id)

    # Initialize states
    ode_state = (msa_init, pair_init)
    gt_msa, gt_pair = msa_init.clone(), pair_init.clone()

    # Integrate Neural ODE
    pred_trajectory = odeint(ode_func, ode_state, t_grid, method='rk4')

    # Accumulate loss over all layers
    total_loss = 0

    for i in range(1, len(t_grid)):
        # Advance Evoformer by one layer to get ground truth
        gt_msa, gt_pair = run_evoformer_block(gt_msa, gt_pair)

        # Get Neural ODE prediction at current layer
        pred_msa, pred_pair = pred_trajectory[i]

        # Compute loss
        loss = loss_fn(pred_msa, gt_msa) + loss_fn(pred_pair, gt_pair)
        total_loss += loss

    return total_loss

def train(input_dir):
    #dataset = get_dataset(input_dir)
    #print(f"Found {len(dataset)} sequences in {input_dir}")

    for epoch in range(epochs):
        epoch_loss = 0
        for protein_id in dataset:
            optimizer.zero_grad()
            loss = train_step(protein_id)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(dataset)}")

# === Example Usage ===
dataset = ["4cue_A", "1fme_A"]  # Replace with your dataset IDs
train(dataset)

