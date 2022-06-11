import subprocess

# import tempfile
from typing import List, TextIO, Union


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
