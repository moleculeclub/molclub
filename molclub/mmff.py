from typing import List, Tuple

from rdkit import Chem  # type: ignore
from rdkit.Chem import rdForceFieldHelpers  # type: ignore

from molclub import utils


def get_mmff_energy(
    mol: Chem.rdchem.Mol,
) -> float:
    utils.mol_has_one_conf(mol)
    mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=0)
    return mmff[0][1]


def get_mmff_trajectory(
    mol: Chem.rdchem.Mol,
    max_iters: int = 200,
) -> Tuple[List[int], List[float]]:
    utils.mol_has_one_conf(mol)
    energies = [get_mmff_energy(mol)]
    for i in range(max_iters):
        mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=1)
        energies.append(mmff[0][1])
    return list(range(len(energies))), energies
