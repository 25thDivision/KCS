"""Print the Python interpreter and library versions in the current environment.

Usage:
    python check_version.py           # show key project libraries + env info
    python check_version.py --all     # also dump every installed package
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import sys

# Libraries actually imported somewhere in this repo. Grouped for readability.
KEY_PACKAGES: dict[str, list[str]] = {
    "Core scientific": [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "networkx",
        "sympy",
    ],
    "Quantum / QEC": [
        "stim",
        "pymatching",
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "qiskit-ibm-provider",
        "qiskit-terra",
        "ldpc",
    ],
    "Machine learning": [
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torchvision",
        "tensorflow",
        "scikit-learn",
    ],
    "IO / utils": [
        "requests",
        "tqdm",
        "pyyaml",
        "h5py",
        "joblib",
    ],
}


def get_version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "— not installed —"


def print_env_info() -> None:
    print("=" * 70)
    print("Environment")
    print("=" * 70)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Platform          : {platform.platform()}")
    print(f"Machine           : {platform.machine()}")
    # torch-specific hardware info if available
    try:
        import torch

        print(f"PyTorch CUDA avail: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version      : {torch.version.cuda}")
            print(f"GPU device(s)     : {torch.cuda.device_count()} "
                  f"({torch.cuda.get_device_name(0)})")
    except ImportError:
        pass
    print()


def print_key_packages() -> None:
    print("=" * 70)
    print("Key project libraries")
    print("=" * 70)
    width = max(len(p) for group in KEY_PACKAGES.values() for p in group)
    for group_name, pkgs in KEY_PACKAGES.items():
        print(f"\n[{group_name}]")
        for pkg in pkgs:
            print(f"  {pkg.ljust(width)}  {get_version(pkg)}")
    print()


def print_all_packages() -> None:
    print("=" * 70)
    print("All installed packages")
    print("=" * 70)
    dists = sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower())
    width = max(len(d.metadata["Name"]) for d in dists)
    for d in dists:
        print(f"  {d.metadata['Name'].ljust(width)}  {d.version}")
    print(f"\nTotal: {len(dists)} packages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="List every installed package (not just the project ones).",
    )
    args = parser.parse_args()

    print_env_info()
    print_key_packages()
    if args.all:
        print_all_packages()


if __name__ == "__main__":
    main()
