# Section 21 — Cross-Problem Integration

## 1. Purpose

Problem 2 must genuinely use the understanding developed in Problem 1.

The project therefore connects the two problems through the same mathematical
scaled dot-product self-attention mechanism.

The relationship is:

```text
Problem 1
    |
    | Mathematical attention
    v
Q = XWQ
K = XWK
V = XWV
    |
    v
S = QK^T / sqrt(dk)
    |
    v
A = softmax(S)
    |
    v
Y = AV
    |
    v
Problem 2
Warehouse Demand Prediction