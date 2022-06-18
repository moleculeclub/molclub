import subprocess
from dataclasses import dataclass

# import tempfile
from typing import List, Optional, TextIO, Union

from rdkit import Chem  # type: ignore

from molclub.compute import compute_utils


def reopt_ensemble():
    pass


@dataclass(init=True, repr=True, slots=True)
class Parameters(compute_utils.Parameters):
    method: str = "gfnff"
    search_intensity: str = "full"
    rmsd_thresh: float = 0.125
    energy_thresh: float = 0.05
    boltzmann_thresh: float = 0.05
    energy_window: float = 10
    solvation: str = "alpb"
    solvent: str = "water"
    num_mtd_cycle: int = 5
    opt_tightness: str = "vtight"
    prop: Optional[str] = None
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
            )  # format
        if self.solvation not in ["alpb", "gbsa"]:
            raise ValueError(f"{self.solvation} not a valid solvation method")
        if self.opt_tightness not in [
            "crude",
            "sloppy",
            "loose",
            "lax",
            "normal",
            "tight",
            "vtight",
            "extreme",
        ]:
            raise ValueError(f"{self.opt_tightness} not a valid tightness")
        if self.prop not in [None, "hess", "reopt", "autoIR"]:
            raise ValueError(f"{self.prop} not a valid property calculation")

    def get_method(self) -> List[str]:
        method_dict = {
            "gfn0-xtb": ["--gfn", "0"],
            "gfn1-xtb": ["--gfn", "1"],
            "gfn2-xtb": ["--gfn", "2"],
            "gfnff": ["--gfnff"],
        }
        return method_dict[self.method]

    # def get_search_settings(self) -> List[str]:
    # search_intensity_dict = {
    #     "full": []
    #     "fast":
    #     "faster":
    #     "fastest":
    # }

    def get_args(self) -> List[str]:
        args = []
        args += self.get_method()

        return args


class crestError(Exception):
    pass


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


def job(
    mol: Chem.Mol,
    params: Parameters,
    job_type: str = "mtd-gc",
    charge: int = 0,
    num_unpaired_electrons: int = 0,
    use_temp_dir: bool = True,
    working_dir: Optional[str] = None,
):
    pass
