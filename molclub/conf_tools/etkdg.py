from typing import List, Optional, Tuple, Union

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem import rdDistGeom, rdMolAlign, rdMolDescriptors  # type: ignore


def default_embed_params() -> rdDistGeom.EmbedParameters:
    """
    Returns an RDKit EmbedParameters object with the latest parameters.

    Returns
    -------
    rdDistGeom.EmbedParameters
        An RDKit EmbedParameters object that sets the parameters for the ETKDG
        algorithm.
    """
    params = rdDistGeom.ETKDGv3()
    params.forceTransAmides = False
    params.useSmallRingTorsions = True

    return params


def rdkit_conf_gen(
    mol: Chem.Mol,
    num_threads: int = 1,
) -> Tuple[List[Chem.Mol], List[float]]:
    """
    A default method for generating conformers using RDKit's ETKDG and MMFF
    algorithms. Generates 100 ETKDG conformers and optimizes them with MMFF.

    Parameters
    ----------
    mol: Chem.Mol
        Input RDKit Mol.
    num_threads: int = 1,
        The number of CPU threads to use.

    Returns
    -------
    Tuple[List[Chem.Mol], List[float]]
        Returns a list of RDKit Mols and their corresponding energies in
        kcal/mol.
    """
    mols = etkdg(mol, 100, num_threads=num_threads)
    mols, energies = opt_mmff(mols, 50, num_threads)

    return mols, energies


def etkdg(
    mol: Chem.Mol,
    num_confs: Union[str, int] = "auto",
    prune_rms_thresh: float = 0.125,
    num_threads: int = 1,
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
) -> List[Chem.Mol]:
    """
    Wrapper for RDKit's ETKDG (Experimental Torsion Knowledge Distance
    Geometry) conformer generator.

    Parameters
    ----------
    mol: Chem.Mol
        Input RDKit Mol.
    num_confs: Union[str, int] = "auto"
        User can either specify the number of conformers to be generated or use
        "auto" to set num_confs based on the number of rotatable bonds.
    prune_rms_thresh: float = 0.05
        The RMSD cutoff to remove similar molecules.
    num_threads: int = 1,
        The number of CPU threads to use.
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
        An RDKit EmbedParameters object that sets the parameters for the ETKDG
        algorithm.

    Returns
    -------
    List[Chem.Mol]
        List of mols with an embedded conformer.
    """
    # input checking
    if isinstance(num_confs, str) and num_confs != "auto":
        raise ValueError(f'num_confs: expected int or "auto", got {num_confs}')
    if isinstance(num_confs, int) and num_confs < 0:
        raise ValueError("num_confs: cannot be negative")
    mol = Chem.AddHs(mol)

    embed_params.numThreads = num_threads
    embed_params.pruneRmsThresh = prune_rms_thresh

    if num_confs == "auto":
        n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        if n_rot_bonds <= 2:
            num_confs = 100
        elif n_rot_bonds > 2 and n_rot_bonds <= 6:
            num_confs = 200
        elif n_rot_bonds > 6 and n_rot_bonds <= 9:
            num_confs = 300
        elif n_rot_bonds > 9:
            num_confs = 500

    rdDistGeom.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        params=embed_params,
    )

    mols = []
    for conformer in mol.GetConformers():
        temp_mol = Chem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conformer, assignId=True)
        mols.append(temp_mol)

    return mols


def opt_mmff(
    input_mols: List[Chem.Mol],
    max_iters: int = 20,
    num_threads: int = 1,
) -> Tuple[List[Chem.Mol], List[float]]:
    """
    Wrapper for RDKit MMFF conformer optimizer.

    Parameters
    ----------
    input_mols: List[Chem.Mol]
        List of RDKit Mols with embedded conformers.
    max_iters: int = 20
        Number of optimizations steps taken by the MMFF optimizer.
    num_threads: int = 1,
        The number of CPU threads to use.

    Returns
    -------
    Tuple[List[Chem.Mol], List[float]]
        Returns a list of RDKit Mols and their corresponding energies in
        kcal/mol.
    """
    mol = Chem.Mol(input_mols[0], quickCopy=True)
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
        temp_mol = Chem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conf, assignId=True)
        mols.append(temp_mol)

    return mols, energies


def opt_xtb():
    raise NotImplementedError


def prune(
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
