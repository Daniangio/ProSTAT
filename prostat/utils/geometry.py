import torch
import numpy as np

def Rx(theta: float) -> np.ndarray:
    return np.array([[ 1, 0           , 0             ],
                     [ 0, np.cos(theta),-np.sin(theta)],
                     [ 0, np.sin(theta), np.cos(theta)]], dtype=np.float32)
  
def Ry(theta: float) -> np.ndarray:
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           ,  1, 0            ],
                   [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
  
def Rz(theta: float) -> np.ndarray:
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            ,   1 ]], dtype=np.float32)

def get_rotation_matrix(phi: float, theta: float, psi: float) -> torch.Tensor:
    """ Get the rotation matrix for a rotation in 3D space by phi, theta and psi

        :param phi: float     | angle to rotate around x axis
        :param psi: float     | angle to rotate around y axis
        :param theta: float   | angle to rotate around z axis
        :return: torch.Tensor | shape (3, 3)
    """
    
    return  torch.from_numpy(Rz(psi) * Ry(theta) * Rx(phi))

def get_bonds(pos: torch.Tensor, bond_idcs: torch.Tensor) -> torch.Tensor:
    """ Compute bond length over specified bond_idcs for every frame in the batch

        :param pos:       torch.Tensor | shape (batch, n_atoms, xyz)
        :param bond_idcs: torch.Tensor | shape (n_bonds, 2)
        :return:          torch.Tensor | shape (batch, n_bonds)
    """

    dist_vectors = pos[:, bond_idcs]
    dist_vectors = dist_vectors[:, :, 1] - dist_vectors[:, :, 0]
    return torch.norm(dist_vectors, dim=2)

def get_angles_from_vectors(b0: torch.Tensor, b1: torch.Tensor, return_cos: bool = False) -> torch.Tensor:
    b0n = torch.norm(b0, dim=2, keepdim=False)
    b1n = torch.norm(b1, dim=2, keepdim=False)
    angles = torch.sum(b0 * b1, axis=-1) / b0n / b1n
    clamped_cos = torch.clamp(angles, min=-1., max=1.)
    if return_cos:
        return clamped_cos
    return torch.arccos(clamped_cos)

def get_angles(pos: torch.Tensor, angle_idcs: torch.Tensor) -> torch.Tensor:
    """ Compute angle values (in radiants) over specified angle_idcs for every frame in the batch

        :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
        :param angle_idcs: torch.Tensor | shape (n_angles, 3)
        :return:           torch.Tensor | shape (batch, n_angles)
    """

    dist_vectors = pos[:, angle_idcs]
    b0 = -1.0 * (dist_vectors[:, :, 1] - dist_vectors[:, :, 0])
    b1 = (dist_vectors[:, :, 2] - dist_vectors[:, :, 1])
    return get_angles_from_vectors(b0, b1)

def get_dihedrals(pos: torch.Tensor, dihedral_idcs: torch.Tensor) -> torch.Tensor:
    """ Compute dihedral values (in radiants) over specified dihedral_idcs for every frame in the batch

        :param pos:        torch.Tensor | shape (batch, n_atoms, xyz)
        :param dihedral_idcs: torch.Tensor | shape (n_dihedrals, 4)
        :return:           torch.Tensor | shape (batch, n_dihedrals)
    """

    p = pos[:, dihedral_idcs]
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / torch.linalg.vector_norm(b1, dim=1, keepdim=True)

    v = b0 - torch.einsum("ijk,ikj->ij", b0, torch.transpose(b1, 1, 2))[..., None] * b1
    w = b2 - torch.einsum("ijk,ikj->ij", b2, torch.transpose(b1, 1, 2))[..., None] * b1

    x = torch.einsum("ijk,ikj->ij", v, torch.transpose(w, 1, 2))
    y = torch.einsum("ijk,ikj->ij", torch.cross(b1, v), torch.transpose(w, 1, 2))

    return torch.atan2(y, x).reshape(-1, dihedral_idcs.shape[0])