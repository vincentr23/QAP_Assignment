import numpy as np
import requests

from .config import RAW_DATA_DIR


def download_data(instance_name: str):
    """Downloads a .dat file from qaplib.mgi.polymtl.ca."""
    file_path = RAW_DATA_DIR / f"{instance_name}.dat"
    if file_path.exists():
        return
    res = requests.get(f"https://qaplib.mgi.polymtl.ca/data.d/{instance_name}.dat")
    res.raise_for_status()
    with open(file_path, "w") as f:
        f.write(res.text)


def make_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("kra30a", "tai40a"):
        download_data(name)


def parse_dat_file(instance_name: str) -> tuple[int, np.ndarray, np.ndarray]:
    """Parses QAPLib instance file and returns n, A, B."""
    with open(RAW_DATA_DIR / f"{instance_name}.dat", "r") as f:
        content = f.read().split()
    data = np.array(content, dtype=int)
    n = data[0]
    A = data[1 : n**2 + 1].reshape((n, n))
    B = data[n**2 + 1 : 2 * n**2 + 1].reshape((n, n))
    return n, A, B
