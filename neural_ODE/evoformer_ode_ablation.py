import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EvoformerODEFuncAblation(nn.Module):
    """
    EvoformerODEFunc with ablation flags.
    Each module can be replaced with a simple linear layer to measure its contribution.
    """

    def __init__(self, c_m, c_z, hidden_dim=64, num_heads=4,
                 ablate_row_attn=False,
                 ablate_col_attn=False,
                 ablate_msa_transition=False,
                 ablate_outer_product=False,
                 ablate_triangle=False,
                 ablate_pair_transition=False,
                 ablate_time_embed=False):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Store ablation flags
        self.ablate_row_attn = ablate_row_attn
        self.ablate_col_attn = ablate_col_attn
        self.ablate_msa_transition = ablate_msa_transition
        self.ablate_outer_product = ablate_outer_product
        self.ablate_triangle = ablate_triangle
        self.ablate_pair_transition = ablate_pair_transition
        self.ablate_time_embed = ablate_time_embed

        # MSA Row Attention with Pair Bias
        if ablate_row_attn:
            self.msa_row_linear = nn.Linear(c_m, c_m)
        else:
            self.msa_row_norm = nn.LayerNorm(c_m)
            self.msa_row_qkv = nn.Linear(c_m, hidden_dim * 3)
            self.msa_row_gate = nn.Linear(c_m, hidden_dim)
            self.msa_row_out = nn.Linear(hidden_dim, c_m)
            self.pair_bias_norm = nn.LayerNorm(c_z)
            self.pair_bias_proj = nn.Linear(c_z, num_heads)

        # MSA Column Attention
        if ablate_col_attn:
            self.msa_col_linear = nn.Linear(c_m, c_m)
        else:
            self.msa_col_norm = nn.LayerNorm(c_m)
            self.msa_col_proj = nn.Linear(c_m, hidden_dim)
            self.msa_col_gate = nn.Linear(c_m, hidden_dim)
            self.msa_col_out = nn.Linear(hidden_dim, c_m)

        # MSA Transition
        if ablate_msa_transition:
            self.msa_trans_linear = nn.Linear(c_m, c_m)
        else:
            self.msa_trans_norm = nn.LayerNorm(c_m)
            self.msa_trans = nn.Sequential(
                nn.Linear(c_m, 4 * c_m),
                nn.ReLU(),
                nn.Linear(4 * c_m, c_m)
            )

        # Outer Product Mean
        if ablate_outer_product:
            self.outer_linear = nn.Linear(c_m, c_z)
        else:
            self.outer_norm = nn.LayerNorm(c_m)
            self.outer_proj_a = nn.Linear(c_m, 32)
            self.outer_proj_b = nn.Linear(c_m, 32)
            self.outer_out = nn.Linear(32, c_z)

        # Triangle Operations
        if ablate_triangle:
            self.tri_linear = nn.Linear(c_z, c_z)
        else:
            self.tri_norm = nn.LayerNorm(c_z)
            self.tri_proj_a = nn.Linear(c_z, hidden_dim)
            self.tri_proj_b = nn.Linear(c_z, hidden_dim)
            self.tri_gate = nn.Linear(c_z, c_z)
            self.tri_out = nn.Linear(hidden_dim, c_z)

        # Pair Transition
        if ablate_pair_transition:
            self.pair_trans_linear = nn.Linear(c_z, c_z)
        else:
            self.pair_trans_norm = nn.LayerNorm(c_z)
            self.pair_trans = nn.Sequential(
                nn.Linear(c_z, 4 * c_z),
                nn.ReLU(),
                nn.Linear(4 * c_z, c_z)
            )

        # Time embedding
        if ablate_time_embed:
            # Use constant mixing factors
            self.register_buffer('const_mix', torch.tensor([0.5, 0.5]))
        else:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2)
            )

    def msa_row_attention(self, m, z):
        """MSA row attention with pair bias (or linear replacement)"""
        if self.ablate_row_attn:
            return self.msa_row_linear(m)

        n_seq, n_res, _ = m.shape
        m_norm = self.msa_row_norm(m)

        qkv = self.msa_row_qkv(m_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(n_seq, n_res, self.num_heads, self.head_dim)
        k = k.view(n_seq, n_res, self.num_heads, self.head_dim)
        v = v.view(n_seq, n_res, self.num_heads, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('sihd,sjhd->sijh', q, k) * scale

        z_norm = self.pair_bias_norm(z)
        pair_bias = self.pair_bias_proj(z_norm)
        attn = attn + pair_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=2)

        out = torch.einsum('sijh,sjhd->sihd', attn, v)
        out = out.reshape(n_seq, n_res, self.hidden_dim)

        gate = torch.sigmoid(self.msa_row_gate(m_norm))
        out = gate * out

        return self.msa_row_out(out)

    def column_attention(self, m):
        """MSA column attention (or linear replacement)"""
        if self.ablate_col_attn:
            return self.msa_col_linear(m)

        n_seq, n_res, _ = m.shape
        m_norm = self.msa_col_norm(m)

        proj = self.msa_col_proj(m_norm)
        gate = torch.sigmoid(self.msa_col_gate(m_norm))

        scale = 1.0 / math.sqrt(n_seq)
        attn = torch.einsum('sir,sjr->ij', proj, proj) * scale
        attn = F.softmax(attn, dim=1)

        out = torch.einsum('ij,sjr->sir', attn, proj)
        out = gate * out

        return self.msa_col_out(out)

    def msa_transition(self, m):
        """MSA transition MLP (or linear replacement)"""
        if self.ablate_msa_transition:
            return self.msa_trans_linear(m)
        return self.msa_trans(self.msa_trans_norm(m))

    def outer_product(self, m):
        """Outer product mean for MSA->pair communication (or linear replacement)"""
        if self.ablate_outer_product:
            # Simple broadcast: average MSA over sequences, then broadcast
            m_avg = m.mean(dim=0)  # [n_res, c_m]
            out = self.outer_linear(m_avg)  # [n_res, c_z]
            return out.unsqueeze(0).expand(m.size(1), -1, -1)  # [n_res, n_res, c_z]

        m_norm = self.outer_norm(m)
        a = self.outer_proj_a(m_norm)  # [n_seq, n_res, 32]
        b = self.outer_proj_b(m_norm)  # [n_seq, n_res, 32]

        outer = torch.einsum('sia,sjb->ijab', a, b) / m.size(0)
        outer = outer.sum(dim=-1)  # [n_res, n_res, 32]

        return self.outer_out(outer)

    def triangle_update(self, z):
        """Triangle multiplicative update (or linear replacement)"""
        if self.ablate_triangle:
            return self.tri_linear(z)

        z_norm = self.tri_norm(z)
        a = self.tri_proj_a(z_norm)
        b = self.tri_proj_b(z_norm)
        gate = torch.sigmoid(self.tri_gate(z_norm))

        triangle_update = torch.einsum('ikd,kjd->ijd', a, b)
        triangle_update = F.layer_norm(triangle_update, [triangle_update.size(-1)])

        return gate * self.tri_out(triangle_update)

    def pair_transition(self, z):
        """Pair transition MLP (or linear replacement)"""
        if self.ablate_pair_transition:
            return self.pair_trans_linear(z)
        return self.pair_trans(self.pair_trans_norm(z))

    def forward(self, t, state):
        """Forward pass with proper Evoformer operations"""
        m, z = state

        # Time embedding
        if self.ablate_time_embed:
            mix_msa = self.const_mix[0]
            mix_pair = self.const_mix[1]
        else:
            t_scalar = torch.tensor([t], device=m.device, dtype=m.dtype)
            t_emb = self.time_mlp(t_scalar.view(1, 1))
            mix_msa, mix_pair = torch.sigmoid(t_emb).chunk(2, dim=-1)

        # MSA Stack
        m_row = self.msa_row_attention(m, z)
        m = m + m_row

        m_col = self.column_attention(m)
        m = m + m_col

        m_trans = self.msa_transition(m)
        m = m + m_trans

        # Communication
        z_outer = self.outer_product(m)
        z = z + z_outer

        # Pair Stack
        z_tri = self.triangle_update(z)
        z = z + z_tri

        z_trans = self.pair_transition(z)
        z = z + z_trans

        # Compute derivatives
        if self.ablate_time_embed:
            dm_dt = (m - state[0]) * mix_msa
            dz_dt = (z - state[1]) * mix_pair
        else:
            dm_dt = (m - state[0]) * mix_msa.view(1, 1, 1)
            dz_dt = (z - state[1]) * mix_pair.view(1, 1, 1)

        return (dm_dt, dz_dt)

    def get_ablation_config(self):
        """Return current ablation configuration as dict"""
        return {
            'ablate_row_attn': self.ablate_row_attn,
            'ablate_col_attn': self.ablate_col_attn,
            'ablate_msa_transition': self.ablate_msa_transition,
            'ablate_outer_product': self.ablate_outer_product,
            'ablate_triangle': self.ablate_triangle,
            'ablate_pair_transition': self.ablate_pair_transition,
            'ablate_time_embed': self.ablate_time_embed,
        }

    @staticmethod
    def get_ablation_name(config):
        """Generate a name for the ablation configuration"""
        ablated = [k.replace('ablate_', '') for k, v in config.items() if v]
        if not ablated:
            return 'full_model'
        return 'no_' + '_'.join(ablated)


# Predefined ablation configurations for easy use
ABLATION_CONFIGS = {
    'full_model': {},
    'no_row_attn': {'ablate_row_attn': True},
    'no_col_attn': {'ablate_col_attn': True},
    'no_msa_transition': {'ablate_msa_transition': True},
    'no_outer_product': {'ablate_outer_product': True},
    'no_triangle': {'ablate_triangle': True},
    'no_pair_transition': {'ablate_pair_transition': True},
    'no_time_embed': {'ablate_time_embed': True},
}


def create_ablation_model(c_m, c_z, hidden_dim=64, num_heads=4, ablation_name='full_model'):
    """Factory function to create model with specified ablation"""
    config = ABLATION_CONFIGS.get(ablation_name, {})
    return EvoformerODEFuncAblation(c_m, c_z, hidden_dim, num_heads, **config)