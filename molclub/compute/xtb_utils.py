import re
import subprocess
from dataclasses import dataclass
from os.path import exists, isdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, TextIO, Union

from rdkit import Chem  # type: ignore

from molclub.compute import compute_utils
from molclub.conf_tools.conf_utils import conf_from_xyz, mol_has_one_conf


class xtbError(Exception):
    """
    An exception for handling failed xtb runs.
    """

    pass


@dataclass(init=True, repr=True, slots=True)
class Result(compute_utils.Result):
    energy_kcal: float = 0.0
    energy_hartree: float = 0.0
    conf: Optional[Chem.rdchem.Conformer] = None
    dipole: Optional[compute_utils.Dipole] = None
    """
    Class for handling calculation results from xtb.

    Attributes
    ----------
    energy_kcal: float = 0.0
        The energy of the molecule in kcal/mol
    energy_hartree: float = 0.0
        The energy of the molecule in hartrees
    conf: Optional[Chem.rdchem.Conformer] = None
        The xyz geometry of the molecule as an RDKit Conformer.
    dipole: Optional[calc.Dipole] = None
        The dipole moments and total dipole of the molecule.
    """

    def extract_results(
        self,
        cwd: str,
        get_dipole: bool = True,
        get_quadrupole: bool = False,
        get_atomic_charges: bool = False,
    ) -> None:
        # TODO: write out stuff to extract quadrupole and atomic charges
        with open(f"{cwd}/xtb.out") as xtb_out:
            lines = xtb_out.readlines()
            for i, line in enumerate(lines):
                if get_quadrupole or get_atomic_charges:
                    raise NotImplementedError
                if "TOTAL ENERGY" in line:
                    self.energy_hartree = float(re.sub("[^0-9,.,-]", "", line))
                    self.energy_kcal = self.energy_hartree * 627.5
                if "molecular dipole:" in line and get_dipole:
                    dipole_info = lines[i + 3].split()
                    self.dipole = compute_utils.Dipole(
                        x=float(dipole_info[1]),
                        y=float(dipole_info[2]),
                        z=float(dipole_info[3]),
                        total=float(dipole_info[4]),
                    )
        if exists(f"{cwd}/xtbopt.xyz"):
            with open(f"{cwd}/xtbopt.xyz") as xtbopt_xyz:
                xyz = xtbopt_xyz.read().split("\n")[2:]
            xyz.remove("")
            self.conf = conf_from_xyz(xyz)


@dataclass(init=True, repr=True, slots=True)
class Parameters(compute_utils.Parameters):
    method: str = "gfn2-xtb"
    scc_iters: int = 250
    geom_iters: int = 0
    solvation: str = "alpb"
    solvent: str = "water"
    electrostatic_potential: bool = False
    orbitals: bool = False
    num_threads: int = 1
    """
    Class for managing the parameters for the xtb run.

    Attributes
    ----------
    method: str = "gfn2-xtb"
        The method used for energy calculation. The available methods are
        semi-empirical (gfn#-xtb where # = 0, 1, 2) and force-field (gfnff).
    scc_iters: int = 250
        Number of iterations allowed for the self-consistent charge (xtb
        version of the self-consistent field).
    solvation: str = "alpb"
        Solvation method, can be alpb or gbsa.
    solvent: str = "water"
        Solvent for solvation. See xtb docs for more information.
    electrostatic_potential: bool = False
        Parameter to calculate electrostatic potential.
    orbitals: bool = False
        Parameter to generate .molden orbital files.
    num_threads: int = 1
        Number of CPU threads to use.
    """

    def __post_init__(self) -> None:
        if self.method not in ["gfn0-xtb", "gfn1-xtb", "gfn2-xtb", "gfnff"]:
            raise ValueError(f"{self.method} not a valid xtb method")
        if self.solvation not in ["alpb", "gbsa"]:
            raise ValueError(f"{self.solvation} not a valid solvation method")

    def get_method(self) -> List[str]:
        method_dict = {
            "gfn0-xtb": ["--gfn", "0"],
            "gfn1-xtb": ["--gfn", "1"],
            "gfn2-xtb": ["--gfn", "2"],
            "gfnff": ["--gfnff"],
        }
        return method_dict[self.method]

    def get_args(self) -> List[str]:
        args = []
        args += self.get_method()
        args += ["--iterations", str(self.scc_iters)]
        if self.geom_iters > 0:
            args += ["--cycles", str(self.geom_iters)]
        args += [f"--{self.solvation}", self.solvent]
        if self.electrostatic_potential:
            args += ["--esp"]
        if self.orbitals:
            args += ["--molden"]
        args += ["--parallel", str(self.num_threads)]

        return args


def job(
    mol: Chem.Mol,
    params: Parameters,
    job_type: str = "sp",
    charge: int = 0,
    num_unpaired_electrons: int = 0,
    use_temp_dir: bool = True,
    working_dir: Optional[str] = None,
) -> Result:
    """
    Wrapper for running an xtb job using user-set parameters on an RDKit Mol.

    Parameters
    ----------
    mol: Chem.Mol
        An RDkit Mol with conformer.
    params: Parameters
        An object for managing the parameters for an xtb run.
    job_type: str = "sp"
        Job type, can be single point (sp), geometry optimization (opt), or
        frequency (freq).
    charge: int = 0
        Charge of the molecule.
    num_unpaired_electrons: int = 0
        Number of unpaired electrons, should be 0 unless working with radicals.
    use_temp_dir: bool = True
        Flag to run calculations in a temporary directory. Must be False to set
        a working directory.
    working_dir: Optional[str] = None
        Path to a working directory. use_temp_dir must be False to set a
        working directory.
    """
    mol_has_one_conf(mol)
    if job_type not in ["sp", "opt", "freq"]:
        raise ValueError(f"{job_type} not a valid job_type")
    if use_temp_dir and working_dir is not None:
        raise ValueError("cannot specify working_dir if use_temp_dir = True")
    if working_dir is not None:
        if working_dir.startswith("~"):
            working_dir = f"{str(Path.home())}/{working_dir[2:]}"

    job_type_dict = {
        "sp": "--scc",
        "opt": "--opt",
        "freq": "--hess",
    }

    xtb_args = [
        "xtb",
        "input.xyz",
        job_type_dict[job_type],
        "--chrg",
        str(charge),
        "--uhf",
        str(num_unpaired_electrons),
    ]
    xtb_args += params.get_args()

    if use_temp_dir:
        with TemporaryDirectory() as tmp:
            Chem.MolToXYZFile(mol, f"{tmp}/input.xyz")
            with open(f"{tmp}/xtb.out", "w") as xtb_out:
                run(xtb_args=xtb_args, cwd=tmp, xtb_out=xtb_out)
            xtb_result = Result()
            xtb_result.extract_results(cwd=tmp)
    else:
        assert isinstance(working_dir, str)
        if not isdir(working_dir):
            raise NotADirectoryError(f"{working_dir} is not a directory")
        else:
            Chem.MolToXYZFile(mol, f"{working_dir}/input.xyz")
            with open(f"{working_dir}/xtb.out", "w") as xtb_out:
                run(
                    xtb_args=xtb_args,
                    cwd=working_dir,
                    xtb_out=xtb_out,
                )
            xtb_result = Result()
            xtb_result.extract_results(cwd=working_dir)

    return xtb_result


def run(
    xtb_args: List[str],
    cwd: str,
    xtb_out: Union[int, TextIO] = subprocess.PIPE,
) -> subprocess.CompletedProcess:
    """
    Wrapper for running xtb through subprocess.run.

    Parameters
    ----------
    xtb_args: List[str]
        A list of arguments for xtb.
    cwd: str
        The directory to run the calculations in.
    xtb_out: Union[int, TextIO] = subprocess.PIPE
        Where to write the output from an xtb run. Can either be
        subprocess.PIPE or an open file with write permission.

    subprocess.CompletedProcess
        Class containing information on the subprocess run.
    """
    proc = subprocess.run(
        xtb_args,
        cwd=cwd,
        stdout=xtb_out,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise xtbError(proc.stderr.decode())

    return proc
