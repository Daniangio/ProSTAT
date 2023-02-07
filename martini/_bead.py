from typing import List
from MDAnalysis.core.groups import Atom


class Bead:

    @property
    def n_atoms(self):
        assert len(self._atoms) == len(self._atom_idcs), \
            f"Bead of type {self.type} has a mismatch between the number of _atoms ({len(self._atoms)})" \
            f"and the number of _atom_idcs ({self._atom_idcs})"
        return len(self._atoms)

    def __init__(self, type: str, bead_atoms: List[str]) -> None:
        self.type: str = type
        self.is_complete: bool = False
        
        self._missing_atoms: List[str] = bead_atoms.copy()
        self._atoms: List[Atom] = []
        self._atom_idcs: List[int] = []
        self._atom_weights: List[float] = []
    
    def is_missing_atom(self, atom_name: str):
        return atom_name in self._missing_atoms
    
    def update(self, atom_name: str, atom: Atom, _id: int):
        assert atom_name in self._missing_atoms, "Trying to update a bead with an atom that does not belong to it or is already present"
        self._atoms.append(atom)
        self._atom_idcs.append(_id)
        self._atom_weights.append(atom.mass)
        self._missing_atoms.remove(atom_name)
        if not self._missing_atoms:
            self.is_complete = True