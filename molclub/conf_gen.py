import time
from typing import List, Optional, Tuple, Union

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import rdDistGeom, rdMolAlign, rdMolDescriptors  # type: ignore


def default_embed_params() -> rdDistGeom.EmbedParameters:
    params = rdDistGeom.ETKDGv3()
    params.forceTransAmides = False
    params.useSmallRingTorsions = True

    return params


def rdkit_conf_gen(
    mol: Chem.rdchem.Mol,
    num_confs: Union[str, int] = "auto",
    prune_rms_thresh: float = 0.05,
    max_iters: int = 20,
    num_threads: int = 1,
) -> Tuple[List[Chem.rdchem.Mol], List[float]]:
    start = time.time()
    mols = etkdg(mol, num_confs, prune_rms_thresh, num_threads)
    step_1 = time.time()
    mols, _ = prune(mols, None, prune_rms_thresh)
    step_2 = time.time()
    mols, energies = opt_mmff(mols, max_iters, num_threads)
    end = time.time()
    print(f"etkdg: {step_1 - start}")
    print(f"prune: {step_2 - step_1}")
    print(f"opt_mmff: {end - step_2}")
    print(f"total: {end - start}")

    return mols, energies


def etkdg(
    mol: Chem.rdchem.Mol,
    num_confs: Union[str, int] = "auto",
    prune_rms_thresh: float = 0.05,
    num_threads: int = 1,
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
) -> List[Chem.rdchem.Mol]:
    # input checking
    if isinstance(num_confs, str) and num_confs != "auto":
        raise ValueError(f'num_confs: expected int or "auto", got {num_confs}')
    if isinstance(num_confs, int) and num_confs < 0:
        raise ValueError("num_confs: cannot be negative")
    mol = Chem.AddHs(mol)

    embed_params.numThreads = num_threads
    embed_params.pruneRmsThresh = prune_rms_thresh
    # if not auto, create user-specified number of conformers
    if num_confs != "auto":
        rdDistGeom.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            params=embed_params,
        )
    # otherwise, generate number of conformers based on number of rotatable
    #  bonds:
    else:
        n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if n_rot_bonds <= 2:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=100,
                params=embed_params,
            )
        elif n_rot_bonds > 2 and n_rot_bonds <= 6:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=200,
                params=embed_params,
            )
        elif n_rot_bonds > 6 and n_rot_bonds <= 9:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=300,
                params=embed_params,
            )
        elif n_rot_bonds > 9:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=500,
                params=embed_params,
            )

    mols = []
    for conformer in mol.GetConformers():
        temp_mol = Chem.rdchem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conformer, assignId=True)
        mols.append(temp_mol)

    return mols


def opt_mmff(
    input_mols: List[Chem.rdchem.Mol],
    max_iters: int = 20,
    num_threads: int = 1,
) -> Tuple[List[Chem.rdchem.Mol], List[float]]:
    mol = Chem.rdchem.Mol(input_mols[0], quickCopy=True)
    for i_mol in input_mols:
        mol.AddConformer(i_mol.GetConformer(), assignId=True)
    mmff = AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=max_iters, numThreads=num_threads
    )
    # energies are in kcal/mol, https://github.com/rdkit/rdkit/issues/3157
    energies = []
    conformers = []
    for i, tup in enumerate(mmff):
        energies.append(tup[1])
        conformers.append(mol.GetConformer(i))
    energies, conformers = zip(*sorted(zip(energies, conformers)))

    mols = []
    for conf in conformers:
        temp_mol = Chem.rdchem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conf, assignId=True)
        mols.append(temp_mol)

    return mols, energies


def opt_xtb():
    raise NotImplementedError


def prune(
    mols: List[Chem.rdchem.Mol],
    energies: Optional[List[float]] = None,
    prune_rms_thresh: float = 0.05,
) -> Tuple[List[Chem.rdchem.Mol], Optional[List[float]]]:
    mols_no_h = [Chem.RemoveHs(mol) for mol in mols]

    remove = [False] * len(mols)
    for i, mol_no_h in enumerate(mols_no_h):
        if remove[i] is False:
            for j, other_mol in enumerate(mols_no_h[i + 1 :]):  # noqa: E203
                rms = rdMolAlign.GetBestRMS(other_mol, mol_no_h)
                if rms < prune_rms_thresh:
                    remove[i + 1 + j] = True

    if energies is None:
        for i in reversed(range(len(remove))):
            if remove[i]:
                mols.pop(i)
    else:
        for i in reversed(range(len(remove))):
            if remove[i]:
                mols.pop(i)
                energies.pop(i)

    return mols, energies
