from typing import List, Tuple

from rdkit import Chem  # type: ignore

from molclub.compute.xtb_utils import Parameters, job
from molclub.conf_tools import conf_utils


def sp(
    mol: Chem.Mol,
    num_threads: int = 1,
) -> float:
    conf_utils.mol_has_one_conf(mol)
    result = job(
        mol,
        Parameters(method="gfnff", num_threads=num_threads),
    )
    return result.energy_kcal


def opt(
    mol: Chem.Mol,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[Chem.Mol, float]:
    conf_utils.mol_has_one_conf(mol)
    result = job(
        mol,
        Parameters(
            method="gfnff", geom_iters=max_iters, num_threads=num_threads
        ),  # format
        job_type="opt",
    )
    mol.RemoveAllConformers()
    mol.AddConformer(result.conf)
    return mol, result.energy_kcal


def opt_traj(
    mol: Chem.Mol,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[List[int], List[float]]:
    conf_utils.mol_has_one_conf(mol)
    energies = [sp(mol)]
    for _ in range(max_iters):
        result = job(
            mol,
            Parameters(method="gfnff", geom_iters=1, num_threads=num_threads),
            job_type="opt",
        )
        mol.RemoveAllConformers()
        mol.AddConformer(result.conf)
        energies.append(result.energy_kcal)
    return list(range(len(energies))), energies
