import subprocess
from dataclasses import dataclass, field
from os.path import isdir
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, TextIO, Union

from rdkit import Chem  # type: ignore

from molclub import compute
from molclub.conf_tools.conf_utils import conf_from_xyz, mol_has_one_conf


class orcaError(Exception):
    """
    An exception for handling failed orca runs.
    """

    pass


@dataclass(init=True, repr=True, slots=True)
class Result(compute.Result):
    energies_kcal: List[float] = field(default_factory=list)
    energy_kcal: float = 0.0
    energy_hartree: float = 0.0
    traj: List[Chem.Conformer] = field(default_factory=list)
    conf: Optional[Chem.Conformer] = None
    dipole: Optional[compute.Dipole] = None
    """
    Class for handling calculation results from xtb.

    Attributes
    ----------
    energy_kcal: float = 0.0
        The energy of the molecule in kcal/mol
    energy_hartree: float = 0.0
        The energy of the molecule in hartrees
    conf: Optional[Chem.Conformer] = None
        The xyz geometry of the molecule as an RDKit Conformer.
    traj: List[Chem.Conformer] = None
        The trajectory traveled during the optimization.
    dipole: Optional[calc.Dipole] = None
        The dipole moments and total dipole of the molecule.
    """

    def extract_results(
        self,
        cwd: str,
    ) -> None:
        with open(f"{cwd}/orca.out") as orca_out:
            get_xyz = False
            xyz = ""
            for line in orca_out.readlines():
                if get_xyz:
                    if line != "---------------------------------\n":
                        xyz += line
                if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                    get_xyz = True
                if line == "\n":
                    get_xyz = False
                    if xyz != "":
                        self.traj.append(conf_from_xyz(xyz.split("\n")[:-2]))
                        xyz = ""
                if "FINAL SINGLE POINT ENERGY" in line:
                    self.energy_hartree = float(line.split()[-1])
                    self.energy_kcal = self.energy_hartree * 627.5
                    self.energies_kcal.append(self.energy_kcal)
        self.conf = self.traj[-1]


@dataclass(init=True, repr=True, slots=True)
class Parameters(compute.Parameters):
    method: str = "r2scan-3c"
    basis_set: str = ""
    scf_iters: int = 50
    geom_iters: int = 20
    solvation: str = "cpcmc"
    solvent: str = "water"
    # orbitals: bool = False
    num_threads: int = 1
    """
    Class for managing the parameters for the xtb run.

    Attributes
    ----------
    method: str = "r2scan-3c"
        The method used for energy calculation, valid options include HF, a DFT
        functional, RI-MP2, DLPNO-MP2, or DLPNO-CCSD(T).
    basis_set: str = ""
        The basis set used for energy calculation.
    scf_iters: int = 50
        Number of iterations allowed for the self-consistent field.
    geom_iters: int = 20
        Number of iterations allowed for the geometry optimization.
    solvation: str = "cpcmc"
        Solvation method, can be alpb or gbsa.
    solvent: str = "water"
        Solvent for solvation. See xtb docs for more information.
    num_threads: int = 1
        Number of CPU threads to use.
    """

    def __post_init__(self) -> None:
        if self.solvation not in ["cpcm", "cpcmc", "smd"]:
            raise ValueError(f"{self.solvation} not a valid solvation method")

    def get_args(self) -> str:
        inp = f"{self.method} {self.basis_set} AUTOAUX\n"
        if self.solvation == "cpcm" or "cpcmc":
            inp += f"!{self.solvation}({self.solvent})\n"
        if self.num_threads > 1:
            inp += f"%PAL NPROCS {self.num_threads} END\n"
        elif self.solvation == "smd":
            inp += f'%CPCM SMD TRUE \nSMDSOLVENT "{self.solvent}"\nEND\n'
        inp += f"%SCF MAXITER {self.scf_iters} END\n"
        inp += f"%GEOM MAXITER {self.geom_iters} END\n"

        return inp


def job(
    mol: Chem.Mol,
    params: Parameters,
    orca_dir: str,
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
        "sp": "!",
        "opt": "!OPT ",
        "freq": "!FREQ ",
    }

    inp = f"{job_type_dict[job_type]}"
    inp += params.get_args()
    inp += f"* xyz {charge} {num_unpaired_electrons + 1}\n"
    inp += "\n".join(Chem.MolToXYZBlock(mol).split("\n")[2:])
    inp += "*"

    if use_temp_dir:
        with TemporaryDirectory() as tmp:
            with open(f"{tmp}/orca.inp", "w") as orca_file:
                orca_file.write(inp)
            with open(f"{tmp}/orca.out", "w") as orca_out:
                run(
                    orca_args=[f"{orca_dir}/orca", "orca.inp"],
                    cwd=tmp,
                    orca_out=orca_out,
                )
            orca_result = Result()
            orca_result.extract_results(cwd=tmp)
    else:
        assert isinstance(working_dir, str)
        if not isdir(working_dir):
            raise NotADirectoryError(f"{working_dir} is not a directory")
        else:
            with open(f"{tmp}/orca.inp", "w") as orca_file:
                orca_file.write(inp)
            with open(f"{working_dir}/orca.out", "w") as orca_out:
                run(
                    orca_args=[f"{orca_dir}/orca", "input.inp"],
                    cwd=working_dir,
                    orca_out=orca_out,
                )
            orca_result = Result()
            orca_result.extract_results(cwd=working_dir)

    return orca_result


def run(
    orca_args: List[str],
    cwd: str,
    orca_out: Union[int, TextIO] = subprocess.PIPE,
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
        orca_args,
        cwd=cwd,
        stdout=orca_out,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise orcaError(proc.stderr.decode())

    return proc
