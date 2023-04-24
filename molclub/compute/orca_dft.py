from typing import List, Tuple

from rdkit import Chem  # type: ignore

from molclub.compute.orca_utils import Parameters, job
from molclub.conf_tools import conf_utils


def sp(
    mol: Chem.Mol,
    orca_dir: str,
    num_unpaired_electrons: int = 0,
    num_threads: int = 1,
) -> float:
    conf_utils.mol_has_one_conf(mol)
    result = job(
        mol,
        Parameters(num_threads=num_threads),
        orca_dir,
        charge=Chem.GetFormalCharge(mol),
        num_unpaired_electrons=num_unpaired_electrons,
    )
    return result.energy_kcal


def opt(
    mol: Chem.Mol,
    orca_dir: str,
    num_unpaired_electrons: int = 0,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[Chem.Mol, float]:
    conf_utils.mol_has_one_conf(mol)
    result = job(
        mol,
        Parameters(geom_iters=max_iters, num_threads=num_threads),
        orca_dir,
        "opt",
        charge=Chem.GetFormalCharge(mol),
        num_unpaired_electrons=num_unpaired_electrons,
    )
    mol.RemoveAllConformers()
    mol.AddConformer(result.conf)
    return mol, result.energy_kcal


def opt_cons(
    mol: Chem.Mol,
    cons_type: List[str],
    cons_atoms: List[List[int]],
    cons_value: List[float],
    orca_dir: str,
    num_unpaired_electrons: int = 0,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[Chem.Mol, float]:
    """
    Parameters
    ----------
    mol: Chem.Mol
    cons_type: List[str]
    cons_atoms: List[List[int]]
    cons_value: List[float]
    orca_dir: str
    num_unpaired_electrons: int = 0
    max_iters: int = 200
    num_threads: int = 1

    Returns
    -------
    Tuple[Chem.Mol, float]
    """
    conf_utils.mol_has_one_conf(mol)
    params = Parameters(
        geom_iters=max_iters,
        cons_type=cons_type,
        cons_atoms=cons_atoms,
        cons_value=cons_value,
        num_threads=num_threads,
    )
    result = job(
        mol,
        params,
        orca_dir,
        "opt",
        charge=Chem.GetFormalCharge(mol),
        num_unpaired_electrons=num_unpaired_electrons,
    )
    mol.RemoveAllConformers()
    mol.AddConformer(result.conf)
    return mol, result.energy_kcal


def opt_traj(
    mol: Chem.Mol,
    orca_dir: str,
    num_unpaired_electrons: int = 0,
    max_iters: int = 200,
    num_threads: int = 1,
) -> Tuple[List[Chem.Mol], List[float]]:
    conf_utils.mol_has_one_conf(mol)
    result = job(
        mol,
        Parameters(geom_iters=max_iters, num_threads=num_threads),
        orca_dir,
        "opt",
        charge=Chem.GetFormalCharge(mol),
        num_unpaired_electrons=num_unpaired_electrons,
    )
    mols = []
    for conf in result.traj:
        temp_mol = Chem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conf)
        mols.append(temp_mol)
    return mols, result.energies_kcal
