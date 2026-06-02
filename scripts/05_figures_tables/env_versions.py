"""
Print versions of Python and selected libraries for the current environment.
Optionally save to a text file.
"""

import sys
import argparse


def safe_import(name: str) -> str:
    try:
        m = __import__(name)
        return getattr(m, "__version__", "installed")
    except Exception:
        return "not installed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="Path to output file (if omitted, only prints to stdout).",
    )
    args = parser.parse_args()

    libs = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "xgboost",
        "autogluon",
        "shap",
        "shapiq",
        "julearn",
        "seaborn",
        "skope-rules"
    ]

    lines = []
    lines.append(f"Python {sys.version.split()[0]}")
    for lib in libs:
        lines.append(f"{lib}: {safe_import(lib)}")

    text = "\n".join(lines)

    # Always print to stdout
    print(text)

    # Optionally also save to file
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
