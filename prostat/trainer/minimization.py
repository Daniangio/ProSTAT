import numpy as np
from typing import Dict, Optional
import torch
from prostat.utils.geometry import get_bonds, get_angles, get_dihedrals


def minimization_step(
        pos: torch.Tensor,                              # (batch, n_atoms, xyz)
        bond_idcs: Optional[torch.Tensor] = None,       # (n_bonds, 2)
        bond_params: Optional[torch.Tensor] = None,     # (None, n_bonds, 2)
        angle_idcs: Optional[torch.Tensor] = None,      # (n_angles, 3)
        angle_params: Optional[torch.Tensor] = None,    # (None, n_angles, 2)
        dihedral_idcs: Optional[torch.Tensor] = None,   # (n_dihedrals, 4)
        dihedral_params: Optional[torch.Tensor] = None, # (None, n_dihedrals, 3)
        lr: float = 1e-4,
    ):
    pos.requires_grad = True
    
    energy_components = []
    if bond_idcs is not None:
        energy_components.append(
            torch.sum(
                0.5 * bond_params[..., 1] * (get_bonds(pos, bond_idcs) - bond_params[..., 0])**2)
            )
    if angle_idcs is not None:
        energy_components.append(
            torch.sum(
                0.5 * angle_params[..., 1] * (get_angles(pos, angle_idcs) - angle_params[..., 0])**2)
            )
    if dihedral_idcs is not None:
        energy_components.append(
            torch.sum(
                dihedral_params[..., 1] * (1 + torch.cos(dihedral_params[..., 2] * get_dihedrals(pos, dihedral_idcs) - dihedral_params[..., 0])))
            )

    grads = torch.autograd.grad(
        energy_components,
        [pos],
        create_graph=False
    )[0].detach()

    return pos.detach() - grads * lr, torch.abs(grads).max(), torch.abs(grads).mean()


def minimization(pos: torch.Tensor, dataset: Dict, conf: Dict):
    device = conf.get('minimization_device', conf.get('device', 'cpu'))
    bond_params: np.ndarray = dataset['bond_params']
    if len(bond_params.shape) == 2:
        bond_params = bond_params.reshape(1, len(bond_params), -1)
    bond_params = torch.from_numpy(bond_params).to(device)

    angle_params: np.ndarray = dataset['angle_params']
    if len(angle_params.shape) == 2:
        angle_params = angle_params.reshape(1, len(angle_params), -1)
    angle_params = torch.from_numpy(angle_params).to(device)

    dihedral_params: np.ndarray = dataset['dihedral_params']
    if len(dihedral_params.shape) == 2:
        dihedral_params = dihedral_params.reshape(1, len(dihedral_params), -1)
    dihedral_params = torch.from_numpy(dihedral_params).to(device)
    
    tolerance_step_1 = conf.get('minimization_tolerance_step_1', 5.)
    tolerance_step_2 = conf.get('minimization_tolerance_step_2', 5.)
    tolerance_step_3 = conf.get('minimization_tolerance_step_3', 5.)

    bond_idcs = dataset['bond_idcs']
    angle_idcs = dataset['angle_idcs']
    dihedral_idcs = dataset['dihedral_idcs']
    step = 1
    print("STEP 1")
    for i in range(conf.get('minimizaiton_max_iter', 100000)):
        if step == 1:
            pos, max_grad, mean_grad = minimization_step(
                pos.to(device),
                bond_idcs,
                bond_params,
                angle_idcs,
                angle_params,
                lr=conf.get('minimization_lr', 1e-5)
            )
            if mean_grad < tolerance_step_1 and max_grad < 10*tolerance_step_1:
                step = 2
                print("STEP 2")
        elif step == 2:
            pos, max_grad, mean_grad = minimization_step(
                pos.to(device),
                dihedral_idcs=dihedral_idcs,
                dihedral_params=dihedral_params,
                lr=conf.get('minimization_lr', 1e-5)
            )
            if mean_grad < tolerance_step_2 and max_grad < 50*tolerance_step_2:
                step = 3
                print("STEP 3")
        elif step == 3:
            pos, max_grad, mean_grad = minimization_step(
                pos.to(device),
                bond_idcs=bond_idcs,
                bond_params=bond_params,
                angle_idcs=angle_idcs,
                angle_params=angle_params,
                # dihedral_idcs,
                # dihedral_params,
                lr=conf.get('minimization_lr', 1e-5)
            )
            if mean_grad < tolerance_step_3 and max_grad < 10*tolerance_step_3:
                return pos.detach().cpu()
        if not i%10:
            print(step, max_grad, mean_grad)
    print(f"Max number of iterations reached ({conf.get('minimizaiton_max_iter', 100000)})")
    return pos.detach().cpu()