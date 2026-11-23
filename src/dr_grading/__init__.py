"""DR-Grading generalization study package.

Modules:
- config: paths, seeds, label scheme, hyperparameters
- preprocessing: fundus crop + geometric / physical-scale normalization
- data: datasets and dataloaders for APTOS (train) and Aizawl (external test)
- integrity_check: Stage 0 data-integrity and distribution reporting
- model: RETFound (ViT-L) backbone + grading head
- metrics: Quadratic Weighted Kappa and bootstrap confidence intervals
"""

__all__ = [
    "config",
    "preprocessing",
    "data",
    "integrity_check",
    "model",
    "metrics",
]
