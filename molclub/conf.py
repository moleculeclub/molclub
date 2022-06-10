from typing import List, Tuple, Union

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import rdDistGeom, rdMolAlign, rdMolDescriptors  # type: ignore


def default_embed_params() -> rdDistGeom.EmbedParameters:
    params = rdDistGeom.ETKDGv3()
    params.forceTransAmides = False
    params.useSmallRingTorsions = True

    return params


def get_conformers(
    mol: Chem.rdchem.Mol,
    num_confs: Union[str, int] = "auto",
    max_iters: int = 200,
    num_threads: int = 1,
    prune_rms_thresh: float = 0.5,
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
):
    # # generate conformers with etkdg
    # mol = etkdg_conf_gen(
    #     mol=mol,
    #     num_confs=num_confs,
    #     max_iters=max_iters,
    #     num_threads=num_threads,
    #     prune_rms_thresh=-1,
    #     embed_params=embed_params,
    # )
    # # optimize all conformers with mmff
    # conformers, energies = mmff_optimization(mol, num_threads=num_threads)
    # # convert energies to relative energies
    # min_e = min(energies)
    # energies = [e - min_e for e in energies]
    # # keep all molecules with energy within 10 kcal/mol
    # mol = Chem.rdchem.Mol(mol, quickCopy=True)
    # for conf, e in zip(conformers, energies):
    #     if e < 10:
    #         mol.AddConformer(conf, assignId=True)
    # # remove conformers with RMSD < prune_rms_thresh
    # mol = prune_conformers(mol, prune_rms_thresh)
    # # convert list of conformers to list of mols
    # mols = []
    # for conf in mol.GetConformers():
    #     temp_mol = Chem.rdchem.Mol(mol, quickCopy=True)
    #     temp_mol.AddConformer(conf)
    #     mols.append(temp_mol)

    mol = etkdg_conf_gen(
        mol=mol,
        num_confs=num_confs,
        max_iters=max_iters,
        num_threads=num_threads,
        prune_rms_thresh=-1,
        embed_params=embed_params,
    )
    conformers, energies = mmff_optimization(
        mol, max_iters=10, num_threads=num_threads
    )  # formatting comment
    min_e = min(energies)
    energies = [e - min_e for e in energies]
    mol = Chem.rdchem.Mol(mol, quickCopy=True)
    for conf, e in zip(conformers, energies):
        if e < 10:
            mol.AddConformer(conf, assignId=True)
    mol = prune_conformers(mol, prune_rms_thresh / 2)
    conformers, energies = mmff_optimization(
        mol,
        num_threads=num_threads,
    )
    min_e = min(energies)
    energies = [e - min_e for e in energies]
    mol = Chem.rdchem.Mol(mol, quickCopy=True)
    for conf, e in zip(conformers, energies):
        if e < 10:
            mol.AddConformer(conf, assignId=True)
    mols = prune_conformers(mol, prune_rms_thresh)

    mols = []
    for conf in mol.GetConformers():
        temp_mol = Chem.rdchem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conf)
        mols.append(temp_mol)

    return mols


def etkdg_conf_gen(
    mol: Chem.rdchem.Mol,
    num_confs: Union[str, int] = "auto",
    max_iters: int = 200,
    num_threads: int = 1,
    prune_rms_thresh: float = 0.5,
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
) -> Chem.rdchem.Mol:
    if isinstance(num_confs, str) and num_confs != "auto":
        raise ValueError(f'num_confs: expected int or "auto", got {num_confs}')
    if isinstance(num_confs, int) and num_confs < 0:
        raise ValueError("num_confs: cannot be negative")

    mol = Chem.AddHs(mol)

    embed_params.maxIterations = max_iters
    embed_params.numThreads = num_threads
    embed_params.pruneRmsThresh = prune_rms_thresh

    if num_confs != "auto":
        rdDistGeom.EmbedMultipleConfs(
            mol,
            numConfs=num_confs,
            params=embed_params,
        )
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
                numConfs=300,
                params=embed_params,
            )
        elif n_rot_bonds > 6 and n_rot_bonds <= 9:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=500,
                params=embed_params,
            )
        elif n_rot_bonds > 9:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=1000,
                params=embed_params,
            )

    return mol


def mmff_optimization(
    mol: Chem.rdchem.Mol,
    max_iters: int = 100,
    num_threads: int = 1,
) -> Tuple[List[float], List[Chem.rdchem.Conformer]]:
    mmff = AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=max_iters, numThreads=num_threads
    )
    energies = []
    conformers = []
    for i, tup in enumerate(mmff):
        energies.append(tup[1])
        conformers.append(mol.GetConformer(i))
    energies, conformers = zip(*sorted(zip(energies, conformers)))

    # energies are in kcal/mol, https://github.com/rdkit/rdkit/issues/3157
    return conformers, energies


def prune_conformers(
    input_mol: Chem.rdchem.Mol,
    prune_rms_thresh: float = 0.5,
) -> Chem.rdchem.Mol:
    if len(input_mol.GetConformers()) == 0:
        raise ValueError("expected >1 conformers, got 0 conformers")
    mols = []
    for conf in input_mol.GetConformers():
        temp_mol = Chem.rdchem.Mol(input_mol, quickCopy=True)
        temp_mol.AddConformer(conf)
        mols.append(temp_mol)
    mols_no_h = [Chem.RemoveHs(mol) for mol in mols]

    remove = [False] * len(mols)
    for i, mol_no_h in enumerate(mols_no_h):
        if remove[i] is False:
            for j, other_mol in enumerate(mols_no_h[i + 1 :]):  # noqa: E203
                rms = rdMolAlign.GetBestRMS(other_mol, mol_no_h)
                if rms < prune_rms_thresh:
                    remove[i + 1 + j] = True

    for i in reversed(range(len(remove))):
        if remove[i]:
            mols.pop(i)

    return_mol = Chem.rdchem.Mol(input_mol, quickCopy=True)
    for i, mol in enumerate(mols):
        return_mol.AddConformer(mol.GetConformer(), assignId=True)

    return return_mol
