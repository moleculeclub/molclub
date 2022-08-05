from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit import Geometry  # type: ignore
from rdkit.Chem import rdMolAlign  # type: ignore


def order_confs(
    mols: List[Chem.Mol],
    energies=[],
    sp_method=None,
    **kwargs,
):
    if len(energies) == 0:
        energies = []
        for mol in mols:
            energies.append(sp_method(mol, **kwargs))
    energies, mols = zip(*sorted(zip(energies, mols)))
    mols = list(mols)
    energies = list(energies)

    return mols, energies


def align_confs(
    mols,
):
    rmsd = [0.0]
    for mol in mols[1:]:
        rmsd.append(rdMolAlign.GetBestRMS(mol, mols[0]))
    return rmsd

def prune(
    mols: List[Chem.Mol],
    energies: Optional[List[float]] = None,
    prune_rms_thresh: float = 0.125,
) -> Tuple[List[Chem.Mol], Optional[List[float]]]:

    def prune_once(
        mols: List[Chem.Mol],
        energies: Optional[List[float]] = None,
        prune_rms_thresh: float = 0.125,
    ) -> Tuple[List[Chem.Mol], Optional[List[float]]]:
        """
        Removes duplicate Mols that have geometry RMSD < prune_rms_thresh.

        Parameters
        ----------
        mols: List[Chem.Mol]
            List of RDKit Mols with embedded conformers.
        energies: Optional[List[float]] = None
            List of the corresponding energies in kcal/mol.
        prune_rms_thresh: float = 0.05
            The RMSD cutoff to remove similar molecules.
        """
        mols_no_h = [Chem.RemoveHs(mol) for mol in mols]

        remove = [False] * len(mols)
        for i, mol_no_h in enumerate(mols_no_h):
            if remove[i] is False:
                for j, other_mol in enumerate(mols_no_h[i + 1 :]):  # noqa: E203
                    rms = rdMolAlign.GetBestRMS(other_mol, mol_no_h)
                    if rms < prune_rms_thresh:
                        # you may wonder why this 1 is here
                        # the j is 0-indexed from the slice [1:]
                        # so we will compare i=o, j=0
                        # however if i=0,j=0 has low RMS, we will set
                        #  remove[0] = True when remove[1] should be True
                        remove[i + j + 1] = True

        if energies is None:
            for i in reversed(range(len(remove))):
                if remove[i]:
                    mols.pop(i)
            return mols
        else:
            for i in reversed(range(len(remove))):
                if remove[i]:
                    mols.pop(i)
                    energies.pop(i)
            return mols, energies

    old_len = len(mols)
    mols, energies = prune_once(mols, energies, prune_rms_thresh)
    new_len = len(mols)
    while new_len != old_len:
        old_len = new_len
        mols, energies = prune_once(mols, energies, prune_rms_thresh)
        new_len = len(mols)
    
    return mols, energies


def boltzmann_pop(energies: List[float], threshold: float = 0.05):
    min_e = min(energies)
    rel_energies = [energy - min_e for energy in energies]

    def boltzmann_ratio(energy: float):
        return 2.7182818284 ** (-energy / (0.00198720425 * 298.15))

    total_pop = 0.0
    ratios = []
    for energy in rel_energies:
        ratio = boltzmann_ratio(energy)
        if total_pop == 0.0 or ratio / total_pop >= threshold:
            total_pop += ratio
            ratios.append(ratio)
        if ratio / total_pop < threshold:
            break

    return [ratio / total_pop for ratio in ratios]


def rmsd_matrix(mols: List[Chem.Mol]):
    mols_no_h = [Chem.RemoveHs(mol) for mol in mols]
    rms_matrix = np.ndarray((len(mols_no_h), len(mols_no_h)))
    rms_matrix.fill(-1)
    for i, mol_1 in enumerate(mols_no_h):
        for j, mol_2 in enumerate(mols_no_h):
            if j >= i:
                rms = rdMolAlign.GetBestRMS(mol_1, mol_2)
                rms_matrix[i][j] = rms
    return pd.DataFrame(rms_matrix)


def conf_from_xyz(
    xyz: List[str],
) -> Chem.Conformer:
    conf = Chem.Conformer(len(xyz))
    for i, line in enumerate(xyz):
        ls = line.split()
        x, y, z = float(ls[1]), float(ls[2]), float(ls[3])
        conf.SetAtomPosition(i, Geometry.rdGeometry.Point3D(x, y, z))
    return conf


def conf_to_xyz(
    conf: Chem.Conformer,
) -> List[str]:
    raise NotImplementedError


def mol_has_one_conf(
    mol: Chem.Mol,
) -> Optional[bool]:
    if len(mol.GetConformers()) != 1:
        raise ValueError(
            "expected RDKit Mol with 1 conformer, got "
            f"{len(mol.GetConformers())} conformers"
        )
    return True
