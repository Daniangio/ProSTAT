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
from martini._singleton import Singleton

class MartiniMapping(metaclass=Singleton):

    _atom2bead: dict[str, str] = {}
    _bead2atom: dict[str, List[str]] = {}

    _incomplete_beads: List[Bead] = []
    _complete_beads: List[Bead] = []
    _max_bead_atoms: int = 0

    _bead2atom_idcs: np.ndarray = None
    _weights: np.ndarray = None

    _atom_positions: np.ndarray = None
    _bead_positions: np.ndarray = None

    _atom_names: np.ndarray = None
    _bead_names: np.ndarray = None

    @property
    def n_atoms(self):
        return len(self._atom_names)

    @property
    def n_beads(self):
        return len(self._bead_names)
    
    @property
    def bead_size(self):
        return self._max_bead_atoms
    
    @property
    def bead2atom_idcs_generator(self):
        for bead2atom_idcs in self._bead2atom_idcs:
            yield bead2atom_idcs[bead2atom_idcs != -1]
    
    @property
    def bead2atom_idcs_mask(self):
        return self._bead2atom_idcs != -1

    def __init__(self, root) -> None:
        self._root = root

        # Iterate configuration files and load all mappings
        # Iterate configuration files and load all mappings
        self._load_mappings()

    def _load_mappings(self):
        for filename in os.listdir(self._root):
            conf = yaml.safe_load(Path(os.path.join(self._root, filename)).read_text())
            mol = conf.get("molecule")

            for atom, bead_settings in conf.get("atoms").items():
                bead = bead_settings.split()[0]
                atom_name = "_".join([mol, atom])
                bead_name = "_".join([mol, bead])
                
                self._atom2bead[atom_name] = bead_name
                
                bead2atom = self._bead2atom.get(bead_name, [])
                assert atom_name not in bead2atom, f"{atom_name} is already present in bead {bead_name}. Duplicate mapping"
                bead2atom.append(atom_name)
                self._bead2atom[bead_name] = bead2atom
    
    def map(self, topology: str, *coordinates):
        # New mapping, clear last records
        self._incomplete_beads.clear()
        self._complete_beads.clear()
        self._max_bead_atoms = 0

        u = mda.Universe(topology, *coordinates)
        return self.map_impl(u)
    
    def map_impl(self, u):
        protein = u.select_atoms('protein')

        # Iterate elements in the system
        atom_names = []
        bead_names = []
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
                _id += 1
        self._atom_names = np.array(atom_names)
        self._bead_names = np.array(bead_names)
        
        # Extract the indices of the atoms in the trajectory file for each bead
        self.compute_bead2atom_idcs_and_weights()
        
        # Read trajectory and map atom coords to bead coords
        atom_positions = []
        bead_positions = []
        for ts in u.trajectory:
            pos = protein.positions
            atom_positions.append(pos)
            weighted_pos = pos[self._bead2atom_idcs] * self._weights[..., None]
            bead_positions.append(np.sum(weighted_pos, axis=1))
        self._atom_positions =  np.stack(atom_positions, axis=0)
        self._bead_positions =  np.stack(bead_positions, axis=0)
    
    def _get_incomplete_bead_from_atom_name(self, atom_name: str):
        bead_type = self._atom2bead[atom_name]
        for bead in self._incomplete_beads:
            if bead.type.__eq__(bead_type) and bead.is_missing_atom(atom_name):
                return bead
        return None
    
    def _update_bead(self, bead: Bead, atom_name: str, atom: Atom, _id: int):
        bead.update(atom_name, atom, _id)
        if bead.is_complete:
            self._complete_beads.append(bead)
            self._incomplete_beads.remove(bead)
    
    def _create_bead(self, atom_name: str, atom: Atom, _id: int):
        bead_type = self._atom2bead[atom_name]
        bead = Bead(bead_type, self._bead2atom[bead_type])
        self._max_bead_atoms = max(self._max_bead_atoms, len(bead._missing_atoms))
        bead.update(atom_name, atom, _id)
        if bead.is_complete:
            self._complete_beads.append(bead)
        else:
            self._incomplete_beads.append(bead)
        return bead_type
    
    def compute_bead2atom_idcs_and_weights(self):
        self._bead2atom_idcs = -np.ones((self.n_beads, self._max_bead_atoms), dtype=int)
        self._weights = np.zeros((self.n_beads, self._max_bead_atoms), dtype=float)
        for i, bead in enumerate(self._complete_beads):
            self._bead2atom_idcs[i, :bead.n_atoms] = np.array(bead._atom_idcs)
            weights = np.array(bead._atom_weights)
            self._weights[i, :bead.n_atoms] = weights / weights.sum()