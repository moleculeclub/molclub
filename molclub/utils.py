from typing import List

from rdkit import Chem  # type: ignore
from rdkit import Geometry  # type: ignore


def conf_from_xyz(
    xyz: List[str],
) -> Chem.rdchem.Conformer:
    conf = Chem.rdchem.Conformer(len(xyz))
    for i, line in enumerate(xyz):
        ls = line.split()
        x, y, z = float(ls[1]), float(ls[2]), float(ls[3])
        conf.SetAtomPosition(i, Geometry.rdGeometry.Point3D(x, y, z))
    return conf
