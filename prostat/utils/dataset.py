from typing import Dict
import numpy as np
import yaml

def load_dataset(conf: Dict):
    dataset: Dict = dict(np.load(conf['dataset_filename'], allow_pickle=True))
    bead_types: np.ndarray = dataset.get('bead_types', None)
    
    # constraints_params = get_constraints_params(conf['constraints_filename'], dataset, bead_types)
    # dataset['constraints_params'] = constraints_params
    # atom masses
    dataset['mass'] = ATOMIC_MASSES[dataset['z']]
    # atom positions
    pos_to_angstrom = conf.get('pos_to_angstrom', 1.)
    energy_to_kcalmol = conf.get('energy_to_kcalmol', 1.)
    dataset['R'] = dataset['R'] * pos_to_angstrom
    # bond equilibrium values
    dataset['bond_params'][:, 0] = dataset['bond_params'][:, 0] * pos_to_angstrom
    # bond force constants
    dataset['bond_params'][:, 1] = dataset['bond_params'][:, 1] * energy_to_kcalmol / pos_to_angstrom**2
    # angle equilibrium values
    dataset['angle_params'][:, 0] = dataset['angle_params'][:, 0] * conf.get('angle_to_rad', 1.)
    # angle force constants
    dataset['angle_params'][:, 1] = dataset['angle_params'][:, 1] * energy_to_kcalmol
    # dihedral phase values
    dataset['dihedral_params'][:, 0] = dataset['dihedral_params'][:, 0] * conf.get('angle_to_rad', 1.)
    # dihedral force constants
    dataset['angle_params'][:, 1] = dataset['angle_params'][:, 1] * energy_to_kcalmol
    # dihedral multiplicities do not need conversion (they are just integers)

    cell = dataset.get("cell", None)
    if conf.get('center_molecule', False) and cell is not None:
        cell = cell * pos_to_angstrom
        R = dataset['R']
        _cell = cell.sum(axis=0)
        hook_atom = R[:, R.shape[1]//2:R.shape[1]//2+1, :]
        shift = hook_atom - _cell/2
        shifted_pos = R - shift
        shifted_pos += (shifted_pos < 0) * _cell
        shifted_pos -= (shifted_pos > _cell) * _cell
        dataset['R'] = shifted_pos

    n_train = conf.get('n_train', int(len(dataset['R']) * 0.8))
    n_valid = conf.get('n_valid', int((len(dataset['R']) - n_train) * 0.5))
    n_test = conf.get('n_test', len(dataset['R']) - n_train - n_valid)
    dataset['pos_train'] = dataset['R'][:n_train]
    dataset['pos_valid'] = dataset['R'][n_train:n_train + n_valid]
    dataset['pos_test'] = dataset['R'][n_train + n_valid:n_train + n_valid + n_test]

    dataset['F'] = dataset['F'] * energy_to_kcalmol / pos_to_angstrom
    dataset['forces_train'] = dataset['F'][:n_train]
    dataset['forces_valid'] = dataset['F'][n_train:n_train + n_valid]
    dataset['forces_test'] = dataset['F'][n_train + n_valid:n_train + n_valid + n_test]

    return dataset

def get_constraints_params(constraints_filename: str, dataset: Dict, bead_types: np.ndarray):
    constraints_params = {}
    
    bead_types = np.array(['_'.join(bt.split('_')[1::2]) for bt in bead_types])
    with open(constraints_filename) as f:
        constraints_dict = yaml.full_load(f)

        bond_idcs = dataset.get('bond_idcs', None)
        bond_param_keys = ["|".join(bt) for bt in bead_types[bond_idcs]]
        try:
            constraints_params["bonds"] = np.array([constraints_dict["bonds"][k] for k in bond_param_keys])
        except Exception as e:
            print(f"{str(e)} Missing parametrization for bonds in constraints file {constraints_filename}")
        
        angle_idcs = dataset.get('angle_idcs', None)
        angle_param_keys = ["|".join(bt) for bt in bead_types[angle_idcs]]
        try:
            constraints_params["angles"] = np.array([constraints_dict["angles"][k] for k in angle_param_keys])
        except Exception as e:
            print(f"{str(e)} Missing parametrization for bonds in constraints file {constraints_filename}")
        
        dihedral_idcs = dataset.get('dihedral_idcs', None)
        # dihedral_param_keys = ["|".join(bt) for bt in bead_types[dihedral_idcs]]
        # try:
        #     constraints_params["dihedrals"] = np.array([constraints_dict["dihedrals"][k] for k in dihedral_param_keys])
        # except Exception as e:
        #     print(f"{str(e)} Missing parametrization for bonds in constraints file {constraints_filename}")
    return constraints_params

ATOMIC_MASSES = np.array([
    1.0,             # X
    1.00782503223,   # H
    4.00260325413,   # He
    7.0160034366,    # Li
    9.012183065,     # Be
    11.00930536,     # B
    12.0000000,      # C
    14.00307400443,  # N
    15.99491461957,  # O
    18.99840316273,  # F
    19.9924401762,   # Ne
    22.9897692820,   # Na
    23.985041697,    # Mg
    26.98153853,     # Al
    27.97692653465,  # Si
    30.97376199842,  # P
    31.9720711744,   # S
    34.968852682,    # Cl
    39.9623831237,   # Ar
    38.9637064864,   # K
    39.962590863,    # Ca
], dtype=np.float32)