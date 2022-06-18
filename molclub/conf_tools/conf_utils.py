from typing import List, Optional

from rdkit import Chem  # type: ignore
from rdkit import Geometry  # type: ignore


def order_confs(
    mols,
    sp_method,
    *args,
):
    energies = []
    for mol in mols:
        energies.append(sp_method(mol, *args))
    energies, mols = zip(*sorted(zip(energies, mols)))

    return mols, energies


def conf_from_xyz(
    xyz: List[str],
) -> Chem.rdchem.Conformer:
    conf = Chem.rdchem.Conformer(len(xyz))
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
