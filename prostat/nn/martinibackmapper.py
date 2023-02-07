from typing import Generator
import torch
from e3nn import o3
import numpy as np


def q_conjugate(q: torch.DoubleTensor):
    scaling = torch.tensor([1, -1, -1, -1], device=q.device)
    return scaling * q

def q_mult(q1, q2):
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.stack((w, x, y, z), -1)

def qv_mult(q1, v2):
    if v2.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {v2.shape}.")
    real_parts = v2.new_zeros(v2.shape[:-1] + (1,))
    q2 = torch.cat((real_parts, v2), -1)
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[..., 1:]


class AELinear(torch.nn.Module):
    def __init__(
        self,
        locality,
        hidden,
        num_out_vectors,
        **kwargs
    ):
        super().__init__()
        self.num_out_vectors = num_out_vectors
        self.locality = locality
        self.hidden = hidden
        weights = torch.nn.Parameter(torch.randn(self.locality, self.hidden, self.num_out_vectors))
        self.register_parameter("weights", weights)        
    
    def forward(self, x):
        return torch.einsum("bijk,ijl->bjlk", x, self.weights)


class MartiniBackmapper(torch.nn.Module):

    def __init__(
        self,
        n_atoms: int,
        n_beads: int,
        bead_size: int,
        bead2atom_idcs_mask: np.ndarray,
        bead2atom_idcs_generator: Generator,
        **kwargs
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_beads = n_beads     # hidden
        self.bead_size = bead_size # locality
        self.register_buffer("bead2atom_idcs_mask", torch.from_numpy(bead2atom_idcs_mask))
        self.bead2atom_idcs_generator = bead2atom_idcs_generator

        p_p1_p2 = 3

        self.cross_prod = o3.TensorProduct(
            f"{self.n_beads}x1o",
            f"{self.n_beads}x1o",
            f"{self.n_beads}x1e",
            [(0, 0, 0, "uuu", False)],
            irrep_normalization='none'
        )

        self.decoder_lin = AELinear(p_p1_p2, self.n_beads, self.bead_size)

        angles_polar = torch.nn.Parameter(torch.randn(self.bead2atom_idcs_mask.sum(), 1))
        self.register_parameter("angles_polar", angles_polar)
        angles_azimutal = torch.nn.Parameter(torch.randn(self.bead2atom_idcs_mask.sum(), 1))
        self.register_parameter("angles_azimutal", angles_azimutal)
        scaling = torch.nn.Parameter(torch.randn(self.n_atoms, 1))
        self.register_parameter("scaling", scaling)

        v_rotated_atoms_num_cumsum = 0
        v_rotated_mask_idcs = torch.zeros((self.n_beads, self.bead2atom_idcs_mask.sum()), dtype=torch.bool)
        for h, bead2atom_idcs_mask in enumerate(self.bead2atom_idcs_mask):
            v_rotated_atoms_num = bead2atom_idcs_mask.sum().item()
            v_rotated_mask_idcs[h, v_rotated_atoms_num_cumsum:v_rotated_atoms_num_cumsum + v_rotated_atoms_num] = True
            v_rotated_atoms_num_cumsum += v_rotated_atoms_num
        self.register_buffer("v_rotated_mask_idcs", v_rotated_mask_idcs)
    
    def decode(self, P, p1, p2, batch):
        device = P.device

        p12 = self.cross_prod(p1.reshape(p1.size(0), -1), p2.reshape(p2.size(0), -1)).reshape(p1.size(0), p1.shape[1], -1)

        stacked = torch.stack([p1, p2, p12], dim=1) # (batch, p1_p2_p12, n_beads, xyz)
        rot_axes = self.decoder_lin(stacked) # (batch, n_beads, bead_size, xyz)
        rot_axes = rot_axes / torch.norm(rot_axes, dim=-1, keepdim=True)
        rot_axes = rot_axes[:, self.bead2atom_idcs_mask].reshape(-1, 3) # (batch * n_atoms_of_beads, 3) | n_atoms_of_beads <= n_beads * bead_size
        q_polar = self.get_quaternions(batch=batch, rot_axes=rot_axes, angles=self.angles_polar) # (batch * n_atoms_of_beads, 4)
        v_ = p1.reshape(-1, 3).repeat_interleave(self.bead2atom_idcs_mask.sum(dim=1).repeat(batch), dim=0)
        v_rotated = qv_mult(q_polar, v_)

        ##############################################

        reconstructed = torch.zeros((batch, self.n_atoms, 3)).to(device)
        for h, (bead2atom_idcs, v_rotated_mask_idcs) in enumerate(zip(self.bead2atom_idcs_generator, self.v_rotated_mask_idcs)):
            batch_v_rotated_mask_idcs = v_rotated_mask_idcs.repeat(batch)
            reconstruction = self.scaling[bead2atom_idcs].repeat(batch, 1) * v_rotated[batch_v_rotated_mask_idcs] + P[:, h].repeat_interleave(v_rotated_mask_idcs.sum(), dim=0)
            reconstructed[:, bead2atom_idcs] += reconstruction.reshape(batch, -1, 3)
            
        return reconstructed, rot_axes

    def get_quaternions(self, batch, rot_axes, angles):
        real_parts = torch.cos(angles)
        real_parts = real_parts.repeat(batch, 1)
        imaginary_parts_multiplier = torch.sin(angles)
        imaginary_parts_multiplier = imaginary_parts_multiplier.repeat(batch, 1)
        imaginary_parts = imaginary_parts_multiplier * rot_axes
        q = torch.cat((real_parts, imaginary_parts), -1)
        return q
    
    def forward(self, P, p1, p2):
        batch = P.size(0)
        reconstructed, p1p2 = self.decode(P, p1, p2, batch)
        
        return reconstructed, p1p2