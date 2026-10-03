"""Section 21: explicitly connect Problem 1 attention to Problem 2."""

from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_warehouse_data
from src.manual_attention import attention_forward


def main():
    out = ROOT / "results" / "cross_problem"
    out.mkdir(parents=True, exist_ok=True)

    df = generate_warehouse_data(n_hours=24 * 14, seed=42)
    window = df["demand"].to_numpy(dtype=np.float64)[:24][None, :, None]

    rng = np.random.default_rng(42)
    WQ = rng.normal(size=(1, 4))
    WK = rng.normal(size=(1, 4))
    WV = rng.normal(size=(1, 4))
    context, cache = attention_forward(window, WQ, WK, WV, scale=True)

    report = {
        "problem_1": "Scaled dot-product self-attention learns weighted relationships among sequence positions.",
        "problem_2": "Warehouse forecasting uses the previous 24 hourly observations to predict the next hour.",
        "shared_mechanism": ["Q=XW_Q", "K=XW_K", "V=XW_V", "QK^T/sqrt(d_k)", "stable softmax", "AV"],
        "dataset_source": "src.data_generator.generate_warehouse_data",
        "demonstration": {
            "input_shape": list(window.shape),
            "attention_shape": list(cache["A"].shape),
            "context_shape": list(context.shape),
            "attention_rows_sum_to_one": bool(np.allclose(cache["A"].sum(axis=-1), 1.0)),
        },
        "interpretation_caution": "Attention weights are diagnostic weighting signals, not causal explanations."
    }
    (out / "section21_cross_problem_report.json").write_text(json.dumps(report, indent=2))
    (out / "section21_cross_problem_report.txt").write_text(
        "Problem 1 provides the mathematical attention mechanism. Problem 2 applies the same equations to the 24-hour warehouse sequence. "
        "The integration demonstration uses the canonical warehouse generator and the authoritative NumPy attention implementation.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
