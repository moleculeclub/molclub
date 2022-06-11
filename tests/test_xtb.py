import warnings
from tempfile import TemporaryFile

from pytest import approx, raises
from rdkit import Chem  # type: ignore
from rdkit.Chem import rdMolAlign  # type: ignore

from molclub import calc, conf_gen, xtb


def test_get_xtb_energy():
    mol = Chem.MolFromSmiles("COC")
    mol = conf_gen.etkdg(mol)[0]
    assert xtb.get_xtb_energy(mol) == approx(-7143, 3)


def test_order_conformers():
    warnings.warn("TODO: write test")
    pass


def test_optimize_xtb():
    mol = Chem.MolFromSmiles("COC")
    mol = conf_gen.etkdg(mol)[0]
    mol, energy = xtb.optimize_xtb(mol)
    assert energy == approx(-7147.597, 0.003)

    test_mol = Chem.MolFromSmiles(
        "[H]C([H])([H])OC([H])([H])[H] |(-1.15137,1.06017,0.434985;-1.16939,0."
        "0993631,-0.0910665;-1.3259,-0.704531,0.636736;-1.97298,0.095162,-0.82"
        "3185;0.0270769,-0.101689,-0.813061;1.17206,-0.109124,0.0128861;1.2854"
        ",0.849117,0.531736;2.02214,-0.27537,-0.643869;1.11295,-0.913096,0.754"
        "838),atomProp:0.isImplicit.1:2.isImplicit.1:3.isImplicit.1:6.isImplic"
        "it.1:7.isImplicit.1:8.isImplicit.1|",
        sanitize=False,
    )
    assert rdMolAlign.GetBestRMS(mol, test_mol) < 0.025


class TestResult:
    def test_extract_results(self):
        xtb_result = xtb.Result()
        xtb_result.extract_results(cwd="./tests/ref_files")
        assert xtb_result.energy_kcal == -3181.555538595963
        assert xtb_result.energy_hartree == -5.070208029635
        assert xtb_result.dipole == calc.Dipole(
            x=-0.017,
            y=-0.9,
            z=-0.0,
            total=2.287,
        )
        # TODO: add more tests


class TestParameters:
    def test_get_args(self):
        params = xtb.Parameters()
        assert params.get_args() == [
            "--gfn",
            "2",
            "--iterations",
            "250",
            "--alpb",
            "water",
            "--parallel",
            "1",
        ]


def test_run():
    with raises(xtb.xtbError):
        xtb.run(
            ["xtb", "meep"],
            "./tests/tmp",
        )
    with TemporaryFile() as tmp:
        proc = xtb.run(
            ["xtb", "--version"],
            "./tests/tmp",
            tmp,
        )
        assert proc.stderr.decode() == "normal termination of xtb\n"


def test_job():
    with raises(ValueError):
        xtb.job(Chem.MolFromSmiles("C"), xtb.Parameters())
    mol = Chem.MolFromSmiles("C")
    mol = conf_gen.etkdg(mol)[0]
    with raises(ValueError):
        xtb.job(mol, xtb.Parameters(), "meep")
    with raises(ValueError):
        xtb.job(mol, xtb.Parameters(), working_dir="./tmp")
