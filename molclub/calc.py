from abc import ABC, abstractmethod
from dataclasses import dataclass


class Parameters(ABC):
    @abstractmethod
    def get_args(self):
        pass


class Result(ABC):
    @abstractmethod
    def extract_results(
        self,
        cwd: str,
        get_dipole: bool = True,
        get_quadrupole: bool = False,
        get_atomic_charges: bool = False,
    ) -> None:
        pass


@dataclass(init=True, repr=True, slots=True, eq=True)
class Dipole:
    x: float
    y: float
    z: float
    total: float
