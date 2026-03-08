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
