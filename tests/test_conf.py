from pytest import approx, raises
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdDistGeom  # type: ignore

from molclub import conf

methane = Chem.MolFromSmiles("C")


def test_default_embed_params():
    assert isinstance(conf.default_embed_params(), rdDistGeom.EmbedParameters)


def test_etkdg_conf_gen():
    with raises(ValueError):
        conf.etkdg_conf_gen(methane, "str")

    with raises(ValueError):
        conf.etkdg_conf_gen(methane, -1)

    mol_1 = Chem.MolFromSmiles("CC")
    mol_1 = conf.etkdg_conf_gen(
        mol_1,
    )
    assert len(mol_1.GetConformers()) == 1

    mol_2 = Chem.MolFromSmiles("C=CC=CC=C")
    mol_2 = conf.etkdg_conf_gen(
        mol_2,
        num_confs=100,
        prune_rms_thresh=1,
    )
    assert len(mol_2.GetConformers()) == 1

    mol_3 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mol_3 = conf.etkdg_conf_gen(
        mol_3,
    )
    assert len(mol_3.GetConformers()) == 6


def test_mmff_optimization():
    mol_1 = Chem.MolFromSmiles("CC")
    mol_1 = conf.etkdg_conf_gen(
        mol_1,
    )
    conformers, energies = conf.mmff_optimization(mol_1)
    assert energies[0] == approx(-4.734365292858474, 0.1)
    # add checker for RMSD of conformer

    mol_2 = Chem.MolFromSmiles("C=CC=CC=C")
    mol_2 = conf.etkdg_conf_gen(
        mol_2,
        prune_rms_thresh=0.1,
    )
    conformers, energies = conf.mmff_optimization(mol_2)
    assert energies[0] == approx(7.7815197115773875, 0.1)
    # add checker for RMSD of conformer

    mol_3 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mol_3 = conf.etkdg_conf_gen(
        mol_3,
    )
    conformers, energies = conf.mmff_optimization(mol_3)
    assert energies[0] == approx(32.0, 1)
    # add checker for RMSD of conformer
