from typing import List, Tuple

from rdkit import Chem  # type: ignore
from rdkit.Chem import rdForceFieldHelpers  # type: ignore

from molclub.conf_tools.conf_utils import mol_has_one_conf

"""
General notes on using RDKit's MMFF:
    There is no method for getting single points so we "optimize" but with
      maxIters=0.
    There is a method called MMFFOptimizeMolecule that looks more appropriate
      for single points, BUT it does not return the energy.
"""


def sp(
    mol: Chem.Mol,
    num_threads: int = 1,
) -> float:
    mol_has_one_conf(mol)
    mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
        mol, maxIters=0, numThreads=num_threads
    )
    return mmff[0][1]


def opt(
    mol: Chem.Mol,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[Chem.Mol, float]:
    mol_has_one_conf(mol)
    mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
        mol, maxIters=max_iters, numThreads=num_threads
    )
    return mol, mmff[0][1]


def opt_traj(
    mol: Chem.Mol,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[List[int], List[float]]:
    mol_has_one_conf(mol)
    energies = [sp(mol)]
    for _ in range(max_iters):
        mmff = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            mol, maxIters=1, numThreads=num_threads
        )
        energies.append(mmff[0][1])
    return list(range(len(energies))), energies
