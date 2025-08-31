import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from openfold.model.heads import MaskedMSAHead, DistogramHead

class EvoformerODEFunc(nn.Module):
    """
    A faithful implementation of Evoformer operations for Neural ODE.
    This version maintains the key architectural ideas while being computationally tractable.
    """

    def __init__(self, c_m, c_z, hidden_dim=64, num_heads=4):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # MSA Row Attention with Pair Bias
        self.msa_row_norm = nn.LayerNorm(c_m)
        self.msa_row_qkv = nn.Linear(c_m, hidden_dim * 3)
        self.msa_row_gate = nn.Linear(c_m, hidden_dim)
        self.msa_row_out = nn.Linear(hidden_dim, c_m)

        # Pair bias for attention
        self.pair_bias_norm = nn.LayerNorm(c_z)
        self.pair_bias_proj = nn.Linear(c_z, num_heads)

        # MSA Column Attention
        self.msa_col_norm = nn.LayerNorm(c_m)
        self.msa_col_proj = nn.Linear(c_m, hidden_dim)
        self.msa_col_gate = nn.Linear(c_m, hidden_dim)
        self.msa_col_out = nn.Linear(hidden_dim, c_m)

        # MSA Transition
        self.msa_trans_norm = nn.LayerNorm(c_m)
        self.msa_trans = nn.Sequential(
            nn.Linear(c_m, 4 * c_m),
            nn.ReLU(),
            nn.Linear(4 * c_m, c_m)
        )

        # Outer Product Mean
        self.outer_norm = nn.LayerNorm(c_m)
        self.outer_proj_a = nn.Linear(c_m, 32)
        self.outer_proj_b = nn.Linear(c_m, 32)
        self.outer_out = nn.Linear(32, c_z)

        # Triangle Operations (maintains geometric reasoning)
        self.tri_norm = nn.LayerNorm(c_z)
        self.tri_proj_a = nn.Linear(c_z, hidden_dim)
        self.tri_proj_b = nn.Linear(c_z, hidden_dim)
        self.tri_gate = nn.Linear(c_z, c_z)
        self.tri_out = nn.Linear(hidden_dim, c_z)

        # Pair Transition
        self.pair_trans_norm = nn.LayerNorm(c_z)
        self.pair_trans = nn.Sequential(
            nn.Linear(c_z, 4 * c_z),
            nn.ReLU(),
            nn.Linear(4 * c_z, c_z)
        )

        # Time embedding for smooth dynamics
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def msa_row_attention(self, m, z):
        """MSA row attention with pair bias"""
        #mixing info across residues for the same sequence
        n_seq, n_res, _ = m.shape
        m_norm = self.msa_row_norm(m) #normalize over channel dimention (c_m). not row dependent

        # QKV projection
        qkv = self.msa_row_qkv(m_norm) #project m_norm to concatenated Q, K, and V vectors, each of size hidden_dim.
        q, k, v = qkv.chunk(3, dim=-1) #split Q, K, V

        # Reshape for multi-head attention - using multiple heads like alphafold does
        q = q.view(n_seq, n_res, self.num_heads, self.head_dim)
        k = k.view(n_seq, n_res, self.num_heads, self.head_dim)
        v = v.view(n_seq, n_res, self.num_heads, self.head_dim)

        # Attention scores
        # multiply q[s, r, h, d] * k[s, k, h, d] element‑wise for matching s, h, d and sum along d
        # s: sequence, r: query res position, k: key res position, h:head, d: embedding dimention
        attn_scores = torch.einsum('srhd,skhd->srhk', q, k) / math.sqrt(self.head_dim)

        # Add pair bias (reshape to fit the attn_scores shape)
        z_norm = self.pair_bias_norm(z) #[n_res, n_res, c_z]. normalized pairwise rep over channel dim
        pair_bias = self.pair_bias_proj(z_norm)  # [n_res, n_res, num_heads]
        pair_bias_expanded = pair_bias.unsqueeze(-0).expand(n_seq, -1, -1, -1)  # [n_seq, n_res, n_res, num_heads]
        pair_bias_expanded = pair_bias_expanded.permute(0, 1, 3, 2)  # [n_seq, n_res, num_heads, n_res]
        attn_scores = attn_scores + pair_bias_expanded

        # Apply softmax and compute output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum('srhk,skhd->srhd', attn_weights, v)
        attn_out = attn_out.contiguous().view(n_seq, n_res, -1) #concatinate the heads

        # Apply gating
        gate = torch.sigmoid(self.msa_row_gate(m_norm))
        out = gate * attn_out

        return self.msa_row_out(out)

    def column_attention(self, m):
        #only mixing info across sequences for a fixed residue position
        #weighted average of sequences rather than complex pattern matching
        """MSA column attention"""
        m_norm = self.msa_col_norm(m)

        # Simple column-wise processing
        m_proj = self.msa_col_proj(m_norm)
        gate = torch.sigmoid(self.msa_col_gate(m_norm))

        # Column-wise attention (weighted average)
        # Transpose to work on columns, then transpose back
        m_t = m_proj.transpose(0, 1)  # [n_res, n_seq, hidden_dim]
        attn_weights = F.softmax(m_t @ m_t.transpose(-2, -1) / math.sqrt(self.hidden_dim), dim=-1)
        m_out = attn_weights @ m_t
        m_out = m_out.transpose(0, 1)  # Back to [n_seq, n_res, hidden_dim]

        return self.msa_col_out(gate * m_out)

    def outer_product(self, m):
        """Outer product mean"""
        m_norm = self.outer_norm(m)
        a = self.outer_proj_a(m_norm)  # [n_seq, n_res, 32]
        b = self.outer_proj_b(m_norm)  # [n_seq, n_res, 32]

        # Element-wise product instead of full outer product
        outer = a.unsqueeze(2) * b.unsqueeze(1)  # [n_seq, n_res, n_res, 32]
        outer_mean = outer.mean(dim=0)  # Average over sequences

        return self.outer_out(outer_mean)

    def triangle_update(self, z):
        """Triangle multiplication that maintains geometric reasoning"""
        #informing of relationships between next nearest neighbors
        z_norm = self.tri_norm(z)

        #a and b are like messages passing along two edges of a triangle
        a = torch.sigmoid(self.tri_proj_a(z_norm))  # [n_res, n_res, hidden_dim]
        b = torch.sigmoid(self.tri_proj_b(z_norm))  # [n_res, n_res, hidden_dim]
        gate = torch.sigmoid(self.tri_gate(z_norm)) # control how much update is applied

        # Modified triangle multiplication
        # i: 1st residue idx, j: 2nd residue idx, k: shared 3rd residue → summed over, d: feature dimension.
        # "If i interacts with k and k interacts with j, then i and j should interact more strongly."
        triangle_update = torch.einsum('ikd,kjd->ijd', a, b)  # Sum over k
        triangle_update = F.layer_norm(triangle_update, [triangle_update.size(-1)]) #renormalize

        out = gate * self.tri_out(triangle_update) #back to pair rep dimentions
        return out

    def forward(self, t, state):
        """Forward pass with proper Evoformer operations"""
        m, z = state

        # Time embedding
        t_scalar = torch.tensor([t], device=m.device, dtype=m.dtype)
        t_emb = self.time_mlp(t_scalar.view(1, 1))
        mix_msa, mix_pair = torch.sigmoid(t_emb).chunk(2, dim=-1)

        # MSA Stack
        m_row = self.msa_row_attention(m, z)
        m = m + m_row

        m_col = self.column_attention(m)
        m = m + m_col

        m_trans = self.msa_trans(self.msa_trans_norm(m))
        m = m + m_trans

        # Communication
        z_outer = self.outer_product(m)
        z = z + z_outer

        # Pair Stack
        z_tri = self.triangle_update(z)
        z = z + z_tri

        z_trans = self.pair_trans(self.pair_trans_norm(z))
        z = z + z_trans

        # Compute derivatives
        dm_dt = (m - state[0]) * mix_msa.view(1, 1, 1)
        dz_dt = (z - state[1]) * mix_pair.view(1, 1, 1)

        return (dm_dt, dz_dt)

class EvoformerODEFunc2(nn.Module):
    """
    Neural ODE dynamics function based on the Evoformer architecture from AlphaFold 2.
    Simplified version with proper dimension handling.
    """

    def __init__(self, c_m, c_z, hidden_dim=128):
        super(EvoformerODEFunc2, self).__init__()
        self.c_m = c_m  # MSA channels
        self.c_z = c_z  # Pair channels
        self.hidden_dim = hidden_dim

        # === MSA STACK ===

        # MSA row-wise attention with pair bias
        self.msa_row_attn = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, c_m),
            nn.ReLU(),
            nn.Linear(c_m, c_m)
        )

        # Pair bias for MSA
        self.pair_to_msa = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_m)
        )

        # MSA column-wise attention
        self.msa_col_attn = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, c_m),
            nn.ReLU(),
            nn.Linear(c_m, c_m)
        )

        # MSA transition (MLP)
        self.msa_transition = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, c_m * 4),
            nn.ReLU(),
            nn.Linear(c_m * 4, c_m)
        )

        # === COMMUNICATION ===

        # Outer product projections
        self.outer_proj_a = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, hidden_dim)
        )
        self.outer_proj_b = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, hidden_dim)
        )
        # Output projection from outer product to pair dimension
        self.outer_to_pair = nn.Linear(hidden_dim ** 2, c_z)

        # === PAIR STACK ===

        # Triangle multiplication (outgoing)
        self.tri_mult_out = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z)
        )

        # Triangle multiplication (incoming)
        self.tri_mult_in = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z)
        )

        # Triangle attention (starting node)
        self.tri_attn_start = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z)
        )

        # Triangle attention (ending node)
        self.tri_attn_end = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, c_z)
        )

        # Pair transition (MLP)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z * 4),
            nn.ReLU(),
            nn.Linear(c_z * 4, c_z)
        )

        # Time embedding for smooth dynamics
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)  # 2 outputs for MSA and pair mixing factors
        )

    def forward(self, t, state):
        """
        Forward pass of the ODE function.

        Args:
            t (torch.Tensor): Current time point (scalar)
            state (tuple): Current state containing (m, z) where:
                - m: MSA representation [N_seq, N_res, c_m]
                - z: Pair representation [N_res, N_res, c_z]

        Returns:
            tuple: Time derivatives (dm/dt, dz/dt)
        """
        m, z = state

        # Print shapes for debugging
        N_seq, N_res, _ = m.shape

        # Time mixing factor: controls how much update to apply
        t_scalar = torch.tensor([t], device=m.device)
        t_emb = self.time_mlp(t_scalar.view(1, 1))
        mix_msa, mix_pair = torch.sigmoid(t_emb).chunk(2, dim=-1)
        mix_msa = mix_msa.view(1, 1, 1)  # Shape for broadcasting
        mix_pair = mix_pair.view(1, 1, 1)  # Shape for broadcasting

        # === MSA STACK OPERATIONS ===

        # 1. MSA Row Attention with Pair Bias
        m_row = self.msa_row_attn(m)

        # Add pair information as bias
        pair_bias = self.pair_to_msa(z)  # [N_res, N_res, c_m]
        # Average along the second dimension to get a per-residue bias
        pair_bias_avg = pair_bias.mean(dim=1)  # [N_res, c_m]
        # Add the bias to each sequence
        m_row = m_row + pair_bias_avg.unsqueeze(0)  # [N_seq, N_res, c_m]

        # 2. MSA Column Attention
        # Transpose for column-wise operation
        m_col_in = m + m_row  # Residual connection
        m_col_t = m_col_in.transpose(0, 1)  # [N_res, N_seq, c_m]
        m_col_out = self.msa_col_attn(m_col_t)
        m_col_out = m_col_out.transpose(0, 1)  # [N_seq, N_res, c_m]

        # 3. MSA Transition (MLP)
        m_trans_in = m_col_in + m_col_out  # Residual connection
        m_trans_out = self.msa_transition(m_trans_in)

        # Final MSA output
        m_out = m_trans_in + m_trans_out  # Residual connection

        # === COMMUNICATION - OUTER PRODUCT MEAN ===

        # Project MSA to lower-dimensional vectors
        a = self.outer_proj_a(m_out)  # [N_seq, N_res, hidden_dim]
        b = self.outer_proj_b(m_out)  # [N_seq, N_res, hidden_dim]

        # Compute outer product mean - careful with dimensions
        # Implementation 1: Direct computation for each position
        outer_products = torch.zeros((N_res, N_res, self.hidden_dim ** 2), device=m.device)

        for i in range(N_res):
            for j in range(N_res):
                # Extract vectors for residues i and j across all sequences
                a_i = a[:, i, :]  # [N_seq, hidden_dim]
                b_j = b[:, j, :]  # [N_seq, hidden_dim]

                # Compute individual outer products
                outer_ij = []
                for s in range(N_seq):
                    a_is = a_i[s].unsqueeze(1)  # [hidden_dim, 1]
                    b_js = b_j[s].unsqueeze(0)  # [1, hidden_dim]
                    outer_is_js = torch.matmul(a_is, b_js)  # [hidden_dim, hidden_dim]
                    outer_ij.append(outer_is_js.flatten())  # [hidden_dim*hidden_dim]

                # Average across sequences
                outer_ij_mean = torch.stack(outer_ij).mean(dim=0)  # [hidden_dim*hidden_dim]
                outer_products[i, j] = outer_ij_mean

        # Project the outer product mean to pair dimensions
        pair_update = self.outer_to_pair(outer_products)  # [N_res, N_res, c_z]

        # === PAIR STACK OPERATIONS ===

        # 1. Triangle Multiplication (Outgoing)
        z_tri_in = z + pair_update  # Residual connection
        z_tri_out = self.tri_mult_out(z_tri_in)

        # 2. Triangle Multiplication (Incoming)
        z_tri_in2 = z_tri_in + z_tri_out  # Residual connection
        z_tri_out2 = self.tri_mult_in(z_tri_in2)

        # 3. Triangle Attention (Starting Node)
        z_attn_in = z_tri_in2 + z_tri_out2  # Residual connection
        z_attn_out = self.tri_attn_start(z_attn_in)

        # 4. Triangle Attention (Ending Node)
        z_attn_in2 = z_attn_in + z_attn_out  # Residual connection
        # Transpose for column-wise operations
        z_attn_t = z_attn_in2.transpose(0, 1)  # [N_res, N_res, c_z]
        z_attn_out2 = self.tri_attn_end(z_attn_t)
        z_attn_out2 = z_attn_out2.transpose(0, 1)  # [N_res, N_res, c_z]

        # 5. Pair Transition (MLP)
        z_trans_in = z_attn_in2 + z_attn_out2  # Residual connection
        z_trans_out = self.pair_transition(z_trans_in)

        # Final pair output
        z_out = z_trans_in + z_trans_out  # Residual connection

        # Compute change rates based on time factor
        dm_dt = (m_out - m) * mix_msa
        dz_dt = (z_out - z) * mix_pair

        return (dm_dt, dz_dt)


class EvoformerODEFuncFast(nn.Module):
    """
    Fast version of the EvoformerODEFunc.
    Simplified architecture with fewer parameters and computations.
    """

    def __init__(self, c_m, c_z, hidden_dim=64):
        super(EvoformerODEFuncFast, self).__init__()
        self.c_m = c_m  # MSA channels
        self.c_z = c_z  # Pair channels
        self.hidden_dim = hidden_dim

        # Single MLP for MSA update
        self.msa_update = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_m)
        )

        # Single MLP for Pair update
        self.pair_update = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_z)
        )

        # Communication from MSA to Pair
        self.msa_to_pair = nn.Sequential(
            nn.LayerNorm(c_m),
            nn.Linear(c_m, c_z)
        )

        # Communication from Pair to MSA
        self.pair_to_msa = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_m)
        )

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, t, state):
        """
        Forward pass of the fast ODE function.

        Args:
            t (torch.Tensor): Current time point (scalar)
            state (tuple): Current state containing (m, z) where:
                - m: MSA representation [N_seq, N_res, c_m]
                - z: Pair representation [N_res, N_res, c_z]

        Returns:
            tuple: Time derivatives (dm/dt, dz/dt)
        """
        m, z = state

        # Time mixing factor
        t_scalar = torch.tensor([t], device=m.device)
        t_emb = self.time_mlp(t_scalar.view(1, 1))
        mix_msa, mix_pair = torch.sigmoid(t_emb).chunk(2, dim=-1)
        mix_msa = mix_msa.view(1, 1, 1)  # Shape for broadcasting
        mix_pair = mix_pair.view(1, 1, 1)  # Shape for broadcasting

        # MSA to Pair communication
        # Average across sequences
        m_avg = m.mean(dim=0, keepdim=True)  # [1, N_res, c_m]
        m_to_z = self.msa_to_pair(m_avg)  # [1, N_res, c_z]

        # Broadcast to pair dimensions
        m_to_z_row = m_to_z.expand(z.size(0), -1, -1)  # [N_res, N_res, c_z]
        m_to_z_col = m_to_z.transpose(0, 1).expand(-1, z.size(1), -1)  # [N_res, N_res, c_z]

        # Add MSA information to pairs
        z_with_msa = z + 0.5 * m_to_z_row + 0.5 * m_to_z_col

        # Pair to MSA communication
        # Average along rows and columns
        z_row_avg = z.mean(dim=1)  # [N_res, c_z]
        z_to_m = self.pair_to_msa(z_row_avg)  # [N_res, c_m]

        # Add pair information to MSA
        m_with_pair = m + z_to_m.unsqueeze(0)  # [N_seq, N_res, c_m]

        # Update MSA and Pair representations
        m_out = self.msa_update(m_with_pair)
        z_out = self.pair_update(z_with_msa)

        # Compute time derivatives with mixing factors
        dm_dt = (m_out - m) * mix_msa
        dz_dt = (z_out - z) * mix_pair

        return (dm_dt, dz_dt)