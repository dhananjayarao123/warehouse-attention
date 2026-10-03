"""
Section 24 — Failure Handling and Debugging

Debugging process:

Initial implementation
        ↓
Unexpected mathematical / gradient result
        ↓
Hypothesis about the bug
        ↓
Diagnostic experiment
        ↓
Correction
        ↓
Verification

Bug investigated:
The scaling factor in scaled dot-product attention must use sqrt(d_k),
where d_k is the projected query/key dimension.

The buggy implementation incorrectly used sqrt(input_dim).
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.manual_attention import attention_backward, attention_forward


# ============================================================
# 1. INITIAL BUGGY IMPLEMENTATION
# ============================================================

def buggy_forward(X, W_Q, W_K, W_V):
    """
    Initial implementation containing the scaling bug.

    BUG:
        Uses sqrt(input_dim)

    Correct:
        Uses sqrt(d_k)
    """

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    raw_scores = Q @ K.T

    # --------------------------------------------------------
    # BUG
    # --------------------------------------------------------
    scores = raw_scores / np.sqrt(X.shape[-1])

    shifted = scores - np.max(
        scores,
        axis=-1,
        keepdims=True
    )

    exp_scores = np.exp(shifted)

    A = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True
    )

    Y = A @ V

    return Y


# ============================================================
# 2. LOSS FUNCTION
# ============================================================

def loss_from_output(Y):
    """
    Simple scalar loss:

        L = 1/2 ||Y||^2
    """

    return 0.5 * np.sum(Y ** 2)


# ============================================================
# 3. NUMERICAL GRADIENT FOR BUGGY IMPLEMENTATION
# ============================================================

def buggy_gradient_by_finite_difference(
    X,
    W_Q,
    W_K,
    W_V,
    eps=1e-6
):
    """
    Compute numerical gradients using central finite differences.

    dL/dtheta ≈
        [L(theta + eps) - L(theta - eps)] / (2 eps)
    """

    gradients = {}

    parameters = {
        "X": X,
        "W_Q": W_Q,
        "W_K": W_K,
        "W_V": W_V,
    }

    for name, parameter in parameters.items():

        gradient = np.zeros_like(parameter)

        for index in np.ndindex(parameter.shape):

            original_value = parameter[index]

            # theta + epsilon
            parameter[index] = original_value + eps

            loss_plus = loss_from_output(
                buggy_forward(
                    X,
                    W_Q,
                    W_K,
                    W_V
                )
            )

            # theta - epsilon
            parameter[index] = original_value - eps

            loss_minus = loss_from_output(
                buggy_forward(
                    X,
                    W_Q,
                    W_K,
                    W_V
                )
            )

            # Restore parameter
            parameter[index] = original_value

            gradient[index] = (
                loss_plus - loss_minus
            ) / (2.0 * eps)

        gradients[name] = gradient

    return gradients


# ============================================================
# 4. FINITE-DIFFERENCE GRADIENT FOR CORRECT IMPLEMENTATION
# ============================================================

def corrected_loss(
    X,
    W_Q,
    W_K,
    W_V
):
    """
    Loss using the corrected first-principles
    attention implementation.
    """

    Y, _ = attention_forward(
        X,
        W_Q,
        W_K,
        W_V,
        scale=True
    )

    return loss_from_output(Y)


def corrected_gradient_by_finite_difference(
    X,
    W_Q,
    W_K,
    W_V,
    parameter_name,
    eps=1e-6
):
    """
    Numerical gradient for the corrected implementation.
    """

    parameters = {
        "X": X,
        "W_Q": W_Q,
        "W_K": W_K,
        "W_V": W_V,
    }

    parameter = parameters[parameter_name]

    gradient = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):

        original_value = parameter[index]

        parameter[index] = original_value + eps

        loss_plus = corrected_loss(
            X,
            W_Q,
            W_K,
            W_V
        )

        parameter[index] = original_value - eps

        loss_minus = corrected_loss(
            X,
            W_Q,
            W_K,
            W_V
        )

        parameter[index] = original_value

        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * eps)

    return gradient


# ============================================================
# 5. MAIN DEBUGGING EXPERIMENT
# ============================================================

def run_debugging_case():

    rng = np.random.default_rng(42)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Deliberately make input_dim != d_k.
    #
    # This exposes the scaling bug.
    # --------------------------------------------------------

    batch_size = 1
    sequence_length = 4

    input_dim = 3
    d_k = 5
    d_v = 5

    X = rng.normal(
        size=(
            batch_size,
            sequence_length,
            input_dim
        )
    )

    W_Q = rng.normal(
        size=(input_dim, d_k)
    )

    W_K = rng.normal(
        size=(input_dim, d_k)
    )

    W_V = rng.normal(
        size=(input_dim, d_v)
    )

    # ========================================================
    # STEP 1 — INITIAL IMPLEMENTATION
    # ========================================================

    print("=" * 70)
    print("SECTION 24 — FAILURE HANDLING AND DEBUGGING")
    print("=" * 70)

    buggy_scale = np.sqrt(input_dim)
    correct_scale = np.sqrt(d_k)

    print("\n[1] INITIAL IMPLEMENTATION")

    print(
        f"Input dimension       = {input_dim}"
    )

    print(
        f"Projected dimension dk = {d_k}"
    )

    print(
        f"Buggy scale           = sqrt({input_dim}) "
        f"= {buggy_scale:.6f}"
    )

    print(
        f"Correct scale         = sqrt({d_k}) "
        f"= {correct_scale:.6f}"
    )

    # ========================================================
    # STEP 2 — OBSERVE THE DIFFERENCE
    # ========================================================

    Y_buggy = buggy_forward(
        X[0],
        W_Q,
        W_K,
        W_V
    )

    Y_correct, correct_cache = attention_forward(
        X,
        W_Q,
        W_K,
        W_V,
        scale=True
    )

    output_difference = np.max(
        np.abs(
            Y_buggy -
            Y_correct[0]
        )
    )

    print("\n[2] DIAGNOSTIC EXPERIMENT")

    print(
        f"Maximum output difference = "
        f"{output_difference:.6e}"
    )

    # ========================================================
    # STEP 3 — HYPOTHESIS
    # ========================================================

    print("\n[3] HYPOTHESIS")

    print(
        "The discrepancy is caused by using "
        "sqrt(input_dim) instead of sqrt(d_k)."
    )

    # ========================================================
    # STEP 4 — INDEPENDENT NUMERICAL CHECK
    # ========================================================

    print("\n[4] NUMERICAL GRADIENT DIAGNOSTIC")

    numerical_corrected = {}

    for parameter_name in (
        "X",
        "W_Q",
        "W_K",
        "W_V"
    ):

        numerical_corrected[
            parameter_name
        ] = corrected_gradient_by_finite_difference(
            X,
            W_Q,
            W_K,
            W_V,
            parameter_name
        )

    # ========================================================
    # STEP 5 — CORRECTED MANUAL BACKWARD PASS
    # ========================================================

    dY = Y_correct.copy()

    manual_gradients = attention_backward(
        dY,
        correct_cache,
        W_Q,
        W_K,
        W_V
    )

    gradient_keys = {
        "X": "dX",
        "W_Q": "dW_Q",
        "W_K": "dW_K",
        "W_V": "dW_V"
    }

    # ========================================================
    # STEP 6 — COMPARE GRADIENTS
    # ========================================================

    print("\n[5] CORRECTED GRADIENT VERIFICATION")

    maximum_errors = {}

    for parameter_name in (
        "X",
        "W_Q",
        "W_K",
        "W_V"
    ):

        manual_gradient = manual_gradients[
            gradient_keys[parameter_name]
        ]

        numerical_gradient = numerical_corrected[
            parameter_name
        ]

        error = np.max(
            np.abs(
                manual_gradient -
                numerical_gradient
            )
        )

        maximum_errors[
            parameter_name
        ] = error

        print(
            f"{parameter_name:>3}: "
            f"max absolute error = "
            f"{error:.6e}"
        )

    # ========================================================
    # STEP 7 — FINAL ASSERTIONS
    # ========================================================

    assert input_dim != d_k

    assert not np.isclose(
        buggy_scale,
        correct_scale
    )

    assert output_difference > 1e-10

    for parameter_name, error in maximum_errors.items():

        assert error < 1e-5, (
            f"Gradient verification failed for "
            f"{parameter_name}: {error}"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n[6] CORRECTION")

    print(
        "Changed:"
    )

    print(
        "    raw_scores / sqrt(input_dim)"
    )

    print(
        "to:"
    )

    print(
        "    raw_scores / sqrt(d_k)"
    )

    print("\n[7] FINAL VERIFICATION")

    print(
        "PASS — corrected attention scaling "
        "uses sqrt(d_k)."
    )

    print(
        "PASS — finite-difference gradients "
        "match the manual backward pass."
    )

    print(
        "PASS — the debugging experiment is "
        "reproducible."
    )

    print("\nDebugging chain:")

    print(
        "Initial implementation"
    )

    print("        ↓")

    print(
        "Unexpected mathematical / gradient result"
    )

    print("        ↓")

    print(
        "Hypothesis: wrong scaling dimension"
    )

    print("        ↓")

    print(
        "Diagnostic: input_dim != d_k"
    )

    print("        ↓")

    print(
        "Correction: divide by sqrt(d_k)"
    )

    print("        ↓")

    print(
        "Finite-difference verification"
    )

    print("        ↓")

    print("PASS")


if __name__ == "__main__":
    run_debugging_case()