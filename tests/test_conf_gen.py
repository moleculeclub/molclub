from pytest import approx, raises  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdDistGeom  # type: ignore

from molclub import conf_gen


def test_default_embed_params():
    assert isinstance(
        conf_gen.default_embed_params(), rdDistGeom.EmbedParameters
    )


def test_rdkit_conf_gen():
    mol_1 = Chem.MolFromSmiles("C=CC=CC=C")
    mols_1, _ = conf_gen.rdkit_conf_gen(mol_1)
    assert len(mols_1) in [7, 8]

    mol_2 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mols_2, _ = conf_gen.rdkit_conf_gen(mol_2)
    assert len(mols_2) in [8]


def test_etkdg():
    methane = Chem.MolFromSmiles("C")
    with raises(ValueError):
        conf_gen.etkdg(methane, "str")
    with raises(ValueError):
        conf_gen.etkdg(methane, -1)

    mol_1 = Chem.MolFromSmiles("CC")
    mols_1 = conf_gen.etkdg(
        mol_1,
    )
    assert len(mols_1) == 1

    mol_2 = Chem.MolFromSmiles("C=CC=CC=C")
    mols_2 = conf_gen.etkdg(
        mol_2,
        num_confs=100,
        prune_rms_thresh=0.5,
    )
    assert len(mols_2) == 2

    mol_3 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mols_3 = conf_gen.etkdg(
        mol_3,
        prune_rms_thresh=0.5,
    )
    assert len(mols_3) == 6


def test_opt_mmff():
    mol_1 = Chem.MolFromSmiles("CC")
    mols_1 = conf_gen.etkdg(
        mol_1,
    )
    mols_1, energies_1 = conf_gen.opt_mmff(mols_1)
    assert energies_1[0] == approx(-4.734365292858474, 0.1)
    # add checker for RMSD of conformer

    mol_2 = Chem.MolFromSmiles("C=CC=CC=C")
    mols_2 = conf_gen.etkdg(
        mol_2,
        prune_rms_thresh=0.1,
    )
    mols_2, energies_2 = conf_gen.opt_mmff(mols_2)
    assert energies_2[0] == approx(7.7815197115773875, 0.1)
    # add checker for RMSD of conformer

    mol_3 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mols_3 = conf_gen.etkdg(
        mol_3,
    )
    mols_3, energies_3 = conf_gen.opt_mmff(mols_3)
    assert energies_3[0] == approx(32.0, 1)
    # add checker for RMSD of conformer


def test_opt_xtb():
    pass


def test_prune():
    mol_1 = Chem.MolFromSmiles("OC(=O)C1=CC=CC=C1C1=CC=CC=N1")
    mols_1 = conf_gen.etkdg(
        mol_1,
    )
    mols_1 = conf_gen.prune(
        mols_1,
        None,
        1,
    )
    assert len(mols_1) == 2

    smi_2 = "NC(=O)C1=CC=C(CN2N=C3NC(=N)NC(=O)C3=C2NC2=CC=CC=C2)C=C1"
    mol_2 = Chem.MolFromSmiles(smi_2)
    mols_2 = conf_gen.etkdg(
        mol_2,
        50,
    )
    mols_2 = conf_gen.prune(mols_2, None, 1)
    assert len(mols_1) == 2
