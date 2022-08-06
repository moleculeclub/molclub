import subprocess
from dataclasses import dataclass, field
from os.path import isdir
from pathlib import Path
from tempfile import TemporaryDirectory

# import tempfile
from typing import List, Optional, TextIO, Union

from rdkit import Chem  # type: ignore

from molclub.compute import compute_utils
from molclub.conf_tools.conf_utils import (
    conf_from_xyz,
    mol_has_one_conf,
    order_confs,
    prune,
)


def reopt_ensemble():
    pass


@dataclass(init=True, repr=True, slots=True)
class Result(compute_utils.Result):
    energies_kcal: List[float] = field(default_factory=list)
    energies_hartree: List[float] = field(default_factory=list)
    confs: List[Chem.Conformer] = field(default_factory=list)

    def extract_results(
        self,
        cwd: str,
    ) -> None:
        # with open(f'{cwd}/crest.out') as crest_out:
        xyz_blocks = []
        with open(f"{cwd}/crest_conformers.xyz") as confs:
            lines = confs.readlines()
            num_atoms = int(lines[0].strip("\n"))
            lines_per_entry = num_atoms + 2
            xyz_block = ""
            for i in range(0, len(lines)):
                xyz_block += lines[i]
                if (i + 1) % lines_per_entry == 0:
                    xyz_blocks.append(xyz_block)
                    xyz_block = ""

        for xyz_block in xyz_blocks:
            xyz_list = xyz_block.split("\n")
            self.energies_hartree.append(float(xyz_list[1]))
            xyz = xyz_list[2:]
            xyz.remove("")
            self.confs.append(conf_from_xyz(xyz))
        self.energies_kcal = [e * 627.5 for e in self.energies_hartree]


@dataclass(init=True, repr=True, slots=True)
class Parameters(compute_utils.Parameters):
    # use_cregen: bool = False
    method: str = "gfnff"
    search_intensity: str = "full"
    # rmsd_thresh: float = 0.125
    # energy_thresh: float = 0.05
    # boltzmann_thresh: float = 0.05
    # energy_window: float = 10
    solvation: str = "alpb"
    solvent: str = "water"
    # num_mtd_cycle: int = 5
    # opt_tightness: str = "vtight"
    # prop: Optional[str] = None
    protonate: bool = False
    deprotonate: bool = False
    tautomerize: bool = False
    taut_iter: int = 2
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.method not in ["gfn0-xtb", "gfn1-xtb", "gfn2-xtb", "gfnff"]:
            raise ValueError(f"{self.method} not a valid xtb method")
        if self.search_intensity not in [
            "full",
            "fast",
            "faster",
            "fastest",
        ]:
            raise ValueError(
                f"{self.search_intensity} not a valid search setting"
            )
        if self.solvation not in ["alpb", "gbsa"]:
            raise ValueError(f"{self.solvation} not a valid solvation method")
        # if self.opt_tightness not in [
        #     "crude",
        #     "sloppy",
        #     "loose",
        #     "lax",
        #     "normal",
        #     "tight",
        #     "vtight",
        #     "extreme",
        # ]:
        #     raise ValueError(f"{self.opt_tightness} not a valid tightness")
        # if self.prop not in [None, "hess", "reopt", "autoIR"]:
        #     raise ValueError(f"{self.prop} not a valid property calculation")
        if self.protonate and self.deprotonate:
            raise ValueError("both protonate and deprotonate cannot be True")

    # def get_use_cregen(self) -> List[str]:
    #     if self.use_cregen:
    #         return ['--cregen']

    def get_method(self) -> List[str]:
        method_dict = {
            "gfn0-xtb": ["--gfn", "0"],
            "gfn1-xtb": ["--gfn", "1"],
            "gfn2-xtb": ["--gfn", "2"],
            "gfnff": ["--gfnff"],
        }
        return method_dict[self.method]

    def get_search_settings(self) -> List[str]:
        search_intensity_dict = {
            "fast": ["--quick"],
            "faster": ["--squick"],
            "fastest": ["--mquick"],
        }

        return search_intensity_dict[self.search_intensity]

    def get_args(self) -> List[str]:
        args = []
        # if self.use_cregen:
        #     args += ['--cregen']
        args += self.get_method()
        if self.search_intensity != "full":
            args += self.get_search_settings()
        args += [f"--{self.solvation}", self.solvent]
        # args += ['--mrest', str(self.num_mtd_cycle)]
        if self.protonate:
            args += ["--protonate"]
        if self.deprotonate:
            args += ["--deprotonate"]
        if self.tautomerize:
            args += ["--tautomerize"]
        args += ["--iter", str(self.taut_iter)]
        args += ["--T", str(self.num_threads)]

        return args


class crestError(Exception):
    pass


def generate_conformers(
    mol: Chem.Mol,
    num_unpaired_electrons: int = 0,
    prune_rms_thresh: float = 0.125,
    crest_params: Parameters = Parameters(),
) -> List[Chem.Mol]:
    mol_has_one_conf(mol)
    result = job(
        mol=mol,
        params=crest_params,
        charge=Chem.GetFormalCharge(mol),
        num_unpaired_electrons=num_unpaired_electrons,
    )
    mols = []
    for conf in result.confs:
        temp_mol = Chem.Mol(mol, quickCopy=True)
        temp_mol.AddConformer(conf, assignId=True)
        mols.append(temp_mol)

    mols, energies = order_confs(mols, energies=result.energies_kcal)
    mols, energies = prune(mols, energies, prune_rms_thresh)

    return mols


def job(
    mol: Chem.Mol,
    params: Parameters,
    charge: int = 0,
    num_unpaired_electrons: int = 0,
    use_temp_dir: bool = True,
    working_dir: Optional[str] = None,
) -> Result:
    mol_has_one_conf(mol)
    if use_temp_dir and working_dir is not None:
        raise ValueError("cannot specify working_dir if use_temp_dir = True")
    if working_dir is not None:
        if working_dir.startswith("~"):
            working_dir = f"{str(Path.home())}/{working_dir[2:]}"

    crest_args = [
        "crest",
        "input.xyz",
        "--chrg",
        str(charge),
        "--uhf",
        str(num_unpaired_electrons),
    ]
    crest_args += params.get_args()

    if use_temp_dir:
        with TemporaryDirectory() as tmp:
            Chem.MolToXYZFile(mol, f"{tmp}/input.xyz")
            with open(f"{tmp}/crest.out", "w") as crest_out:
                run(crest_args=crest_args, cwd=tmp, crest_out=crest_out)
            crest_result = Result()
            crest_result.extract_results(cwd=tmp)
    else:
        assert isinstance(working_dir, str)
        if not isdir(working_dir):
            raise NotADirectoryError(f"{working_dir} is not a directory")
        else:
            Chem.MolToXYZFile(mol, f"{working_dir}/input.xyz")
            with open(f"{working_dir}/crest.out", "w") as crest_out:
                run(
                    crest_args=crest_args,
                    cwd=working_dir,
                    crest_out=crest_out,
                )
            crest_result = Result()
            crest_result.extract_results(cwd=working_dir)

    return crest_result


def run(
    crest_args: List[str],
    cwd: str,
    crest_out: Union[int, TextIO] = subprocess.PIPE,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        crest_args,
        cwd=cwd,
        stdout=crest_out,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise crestError(proc.stderr.decode())

    return proc
