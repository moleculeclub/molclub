from typing import List, Union

from rdkit import Chem  # type: ignore
from rdkit.Chem import rdDistGeom, rdMolDescriptors  # type: ignore

from molclub.compute import mmff
from molclub.conf_tools.conf_utils import order_confs, prune


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


def generate_conformers(
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
    # embed_params.pruneRmsThresh = prune_rms_thresh

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

    mols, energies = order_confs(mols, mmff.sp, num_threads=num_threads)
    mols, energies = prune(mols, energies, prune_rms_thresh=prune_rms_thresh)

    return mols
