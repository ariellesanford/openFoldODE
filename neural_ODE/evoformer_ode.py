import torch
import torch.nn as nn

class EvoformerODEFunc(nn.Module):
    def __init__(self, c_m, c_z, hidden_dim=128):
        super(EvoformerODEFunc, self).__init__()

        # MSA path (simplified attention block)
        self.m_mlp = nn.Sequential(
            nn.Linear(c_m + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_m)
        )

        # Pair path (triangle update block)
        self.z_mlp = nn.Sequential(
            nn.Linear(c_z + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_z)
        )

        # Cross-stream mixing
        self.m_to_z = nn.Linear(c_m, c_z)
        self.z_to_m = nn.Linear(c_z, c_m)

    def forward(self, t, state):
        m, z = state

        # Include time as a feature for M
        t_feature_m = torch.ones_like(m[..., :1]) * t  # (516, 28, 1)
        m_aug = torch.cat([m, t_feature_m], dim=-1)  # (516, 28, 257)

        # Include time as a feature for Z
        t_feature_z = torch.ones_like(z[..., :1]) * t  # (28, 28, 1)
        z_aug = torch.cat([z, t_feature_z], dim=-1)  # (28, 28, 129)

        # Apply MSA path
        m_out = self.m_mlp(m_aug)

        # Apply Pair-to-MSA path
        z_to_m_output = self.z_to_m(z.mean(dim=1))  # (28, 256)
        z_to_m_output = z_to_m_output.unsqueeze(0).expand(m.size(0), -1, -1)  # (516, 28, 256)

        # Combine MSA path with pair-to-MSA
        m_out = m_out + z_to_m_output

        # Apply Pair path
        z_out = self.z_mlp(z_aug)

        # Apply MSA-to-Pair path
        m_to_z_output = self.m_to_z(m.mean(dim=0))  # (28, 128)
        m_to_z_output = m_to_z_output.unsqueeze(0).expand(z.size(0), -1, -1)  # (28, 28, 128)

        # Combine Pair path with msa-to-pair
        z_out = z_out + m_to_z_output

        return (m_out, z_out)
