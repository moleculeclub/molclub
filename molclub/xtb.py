import subprocess


class xtbError(Exception):
    """
    An exception for handling failed xtb runs.
    """
    pass


def xtb_installed() -> bool:
    """
    Check if xtb is installed.
    """
    proc = subprocess.run(
        ["xtb", "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return proc.returncode == 0
