from typing import List, Tuple, Union

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, rdDistGeom, rdMolDescriptors  # type: ignore


def default_embed_params() -> rdDistGeom.EmbedParameters:
    params = rdDistGeom.ETKDGv3()
    params.forceTransAmides = False
    params.useSmallRingTorsions = True

    return params


def etkdg_conf_gen(
    mol: Chem.rdchem.Mol,
    num_confs: Union[str, int] = "auto",
    max_iterations: int = 200,
    num_threads: int = 1,
    prune_rms_thresh: float = 0.5,
    embed_params: rdDistGeom.EmbedParameters = default_embed_params(),
):
    if isinstance(num_confs, str) and num_confs != "auto":
        raise ValueError(f'num_confs: expected int or "auto", got {num_confs}')
    if isinstance(num_confs, int) and num_confs < 0:
        raise ValueError("num_confs: cannot be negative")

    mol = Chem.AddHs(mol)

    embed_params.maxIterations = max_iterations
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
        if n_rot_bonds <= 3:
            rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=100,
                params=embed_params,
            )
        elif n_rot_bonds > 3 and n_rot_bonds <= 6:
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

    return mol


def mmff_optimization(
    mol: Chem.rdchem.Mol,
    max_iters=100,
) -> Tuple[List[float], List[Chem.rdchem.Conformer]]:
    mmff = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters)
    energies = []
    conformers = []
    for i, tup in enumerate(mmff):
        energies.append(tup[1])
        conformers.append(mol.GetConformer(i))
    energies, conformers = zip(*sorted(zip(energies, conformers)))

    # energies are in kcal/mol, https://github.com/rdkit/rdkit/issues/3157
    return energies, conformers


def prune_conformers():
    raise NotImplementedError()
