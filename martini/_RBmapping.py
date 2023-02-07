import os
import sys
import traceback
import yaml
import numpy as np

from pathlib import Path
from typing import List

import MDAnalysis as mda
from MDAnalysis.core.groups import Atom

from martini._bead import Bead
from martini._mapping import MartiniMapping

class RBMartiniMapping(MartiniMapping):

    _bead2atom_of_bead_v1: dict[str, str] = {}
    _bead2atom_of_bead_v2: dict[str, str] = {}

    _bead_v1: np.ndarray = None
    _bead_v2: np.ndarray = None

    def __init__(self, root) -> None:
        super().__init__(root)

    def _load_mappings(self):
        for filename in os.listdir(self._root):
            conf = yaml.safe_load(Path(os.path.join(self._root, filename)).read_text())
            mol = conf.get("molecule")

            for atom, bead_settings in conf.get("atoms").items():
                bead_splits = bead_settings.split()
                bead = bead_splits[0]
                atom_name = "_".join([mol, atom])
                bead_name = "_".join([mol, bead])
                
                self._atom2bead[atom_name] = bead_name
                if len(bead_splits) > 1:
                    versor = bead_splits[1]
                    if versor == "V1":
                        self._bead2atom_of_bead_v1[bead_name] = atom_name
                    elif versor == "V2":
                        self._bead2atom_of_bead_v2[bead_name] = atom_name
                
                bead2atom = self._bead2atom.get(bead_name, [])
                assert atom_name not in bead2atom, f"{atom_name} is already present in bead {bead_name}. Duplicate mapping"
                bead2atom.append(atom_name)
                self._bead2atom[bead_name] = bead2atom
    
    def map_impl(self, u):
        protein = u.select_atoms('protein')

        # Iterate elements in the system
        atom_names = []
        bead_names = []

        bead_v1_atom_idcs = []
        bead_v2_atom_idcs = []
        _id = 0
        for residue in u.residues:
            for atom in residue.atoms:
                mol = residue.resname
                atom_name = "_".join([mol, atom.name])
                atom_names.append(atom_name)

                try:
                    incomplete_bead = self._get_incomplete_bead_from_atom_name(atom_name)
                    if incomplete_bead is not None:
                        self._update_bead(incomplete_bead, atom_name, atom, _id)
                    else:
                        bead_type = self._create_bead(atom_name, atom, _id)
                        bead_names.append(bead_type)
                except AssertionError:
                    _, _, tb = sys.exc_info()
                    traceback.print_tb(tb) # Fixed format
                    tb_info = traceback.extract_tb(tb)
                    filename, line, func, text = tb_info[-1]

                    print('An error occurred on line {} in statement {}'.format(line, text))
                except:
                    pass # print(f"Missing {atom_name} in mapping file")
                if atom_name in self._bead2atom_of_bead_v1.values():
                    bead_v1_atom_idcs.append(_id)
                if atom_name in self._bead2atom_of_bead_v2.values():
                    bead_v2_atom_idcs.append(_id)
                _id += 1
        self._atom_names = np.array(atom_names)
        self._bead_names = np.array(bead_names)

        bead_v1_atom_idcs = np.array(bead_v1_atom_idcs)
        bead_v2_atom_idcs = np.array(bead_v2_atom_idcs)
        
        # Extract the indices of the atoms in the trajectory file for each bead
        self.compute_bead2atom_idcs_and_weights()
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        bead_positions = []

        bead_v1 = []
        bead_v2 = []
        for ts in u.trajectory:
            pos = protein.positions
            atom_positions.append(pos)
            weighted_pos = pos[self._bead2atom_idcs] * self._weights[..., None]
            bead_positions.append(np.sum(weighted_pos, axis=1))
            bead_v1.append(pos[bead_v1_atom_idcs])
            bead_v2.append(pos[bead_v2_atom_idcs])
        self._atom_positions =  np.stack(atom_positions, axis=0)
        self._bead_positions =  np.stack(bead_positions, axis=0)

        bead_v1 = np.stack(bead_v1, axis=0)
        bead_v2 = np.stack(bead_v2, axis=0)

        self._bead_v1 = bead_v1 - self._bead_positions
        self._bead_v2 = bead_v2 - self._bead_positions