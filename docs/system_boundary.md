1. Actors
   - Candidate / ML Engineer
   - Experiment Runner
   - Reviewer

2. System Boundary

   ┌─────────────────────────────────────┐
   │      Warehouse Attention System     │
   │                                     │
   │ Synthetic Data Generator            │
   │        ↓                            │
   │ Dataset Preparation                 │
   │        ↓                            │
   │ Attention Model                     │
   │        ↓                            │
   │ Evaluation / Experiments            │
   └─────────────────────────────────────┘
             │
             ▼
        Experiment Results

   No external warehouse
   No customer data
   No employee data
   No production integration