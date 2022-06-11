from pytest import approx, raises
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdDistGeom  # type: ignore

from molclub import conf_gen

methane = Chem.MolFromSmiles("C")


def test_default_embed_params():
    assert isinstance(conf_gen.default_embed_params(), rdDistGeom.EmbedParameters)


def test_rdkit_conf_gen():
    pass


def test_etkdg():
    # checks input handling
    with raises(ValueError):
        conf_gen.etkdg(methane, "str")
    with raises(ValueError):
        conf_gen.etkdg(methane, -1)

    # ethane, should only have one conformer
    mol = Chem.MolFromSmiles("CC")
    mols = conf_gen.etkdg(
        mol,
    )
    assert len(mols) == 1

    mol = Chem.MolFromSmiles("C=CC=CC=C")
    mols = conf_gen.etkdg(
        mol,
        num_confs=100,
        prune_rms_thresh=0.5,
    )
    assert len(mols) == 2

    mol = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mols = conf_gen.etkdg(
        mol,
        prune_rms_thresh=0.5,
    )
    assert len(mols) == 6


def test_opt_mmff():
    mol = Chem.MolFromSmiles("CC")
    mols = conf_gen.etkdg(
        mol,
    )
    mols, energies = conf_gen.opt_mmff(mols)
    assert energies[0] == approx(-4.734365292858474, 0.1)
    # add checker for RMSD of conformer

    mol = Chem.MolFromSmiles("C=CC=CC=C")
    mols = conf_gen.etkdg(
        mol,
        prune_rms_thresh=0.1,
    )
    mols, energies = conf_gen.opt_mmff(mols)
    assert energies[0] == approx(7.7815197115773875, 0.1)
    # add checker for RMSD of conformer

    mol = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mol = conf_gen.etkdg(
        mol,
    )
    mols, energies = conf_gen.opt_mmff(mols)
    assert energies[0] == approx(32.0, 1)
    # add checker for RMSD of conformer


def test_opt_xtb():
    pass


def test_prune():
    pass
