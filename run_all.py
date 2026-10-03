"""Run the reproducible test and experiment sequence."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

COMMANDS = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "experiments/toy_experiment.py"],
    [sys.executable, "experiments/gradient_comparison.py"],
    [sys.executable, "experiments/warehouse_experiment.py"],
    [sys.executable, "experiments/baseline_comparison.py"],
    [sys.executable, "experiments/ablation.py"],
    [sys.executable, "experiments/training_dynamics.py"],
    [sys.executable, "experiments/distribution_shift.py"],
    [sys.executable, "experiments/failure_analysis.py"],
]

for command in COMMANDS:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
