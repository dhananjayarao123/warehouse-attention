# Section 18 - Evaluation Methodology

## 1. Objective

The warehouse demand forecasting problem is evaluated using a defined
training set, validation set, and test set.

Because warehouse demand is time-dependent, the split is chronological.

Random splitting is not used.

---

# 2. Dataset Split

The complete synthetic warehouse time series is divided as follows:

| Set | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The chronological order is:

```text
Past                                             Future
│                                                  │
▼                                                  ▼

|----------------------|---------------|---------------|
       Training             Validation        Test
         70%                  15%             15%