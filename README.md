# Reject Inference under Class Imbalance with Label Spreading

Graph-based reject inference with Label Spreading under class imbalance, using LendingClub data. Compares a standard pipeline (random sampling + SMOTE) with a neighborhood-aware one (cluster-stratified sampling + KMeans-SMOTE) across five classifiers.

This repository contains the code and results of the master's thesis **"Machine Learning for Credit Risk Modelling: A Framework Addressing Reject Inference and Class Imbalance"** (NOVA Information Management School, Universidade Nova de Lisboa, 2026), supervised by Professor Afshin Ashofteh.

## Overview

Credit scoring models are usually trained only on accepted applicants, because rejected applicants have no repayment outcome. This introduces sample selection bias, and the accepted data are also class-imbalanced (13.1% defaults in LendingClub). Reject inference uses the rejected applicants as unlabeled data, but in practice only a few variables are recorded for both groups. Here, only four application-time features are shared: loan amount, debt-to-income ratio, employment length, and state.

The framework compares two end-to-end pipelines that are identical except for two steps:

| Step | Standard pipeline (STD) | Neighborhood-aware pipeline (NH) |
|---|---|---|
| Sampling of rejected applicants | Random sampling | Cluster-stratified sampling (k-means strata, proportional allocation) |
| Oversampling of accepted defaults | SMOTE | KMeans-SMOTE (Douzas, Bação & Last, 2018) |
| Label inference | Label Spreading, RBF kernel (γ = 2, α = 0.5) | Same |
| Classifiers | LR, RF, GBDT, XGBoost, MLP | Same |
| Evaluation | ROC-AUC, PR-AUC, Brier score on a 20% accepted test set, 20 seeds, paired Wilcoxon tests | Same |

## Main findings

- NH produces more confident and more differentiated inferred labels for rejected applicants (mean confidence 0.78 vs 0.70 at γ = 2).
- The effect on prediction depends on the classifier: NH improves ROC-AUC and PR-AUC for logistic regression and the MLP, while STD performs better for the tree ensembles.
- NH achieves lower Brier scores for most non-linear classifiers. All models remain above the Brier score of a constant prediction at the observed default rate (about 0.114), so these comparisons are relative.
- An accepted-only baseline shows that adding inferred rejected applicants does not improve ranking on the accepted test set.

See the thesis for the full results, limitations, and discussion.

## Repository structure

```
├── notebooks/
│   └── reject_inference_pipeline.ipynb   # full pipeline: data preparation, experiments, tests
├── results/
│   ├── main/            # main experiment (N_L = 10,000, S = 3,000, γ = 2, 20 seeds)
│   ├── gamma_sweep/     # label propagation across γ (10 seeds)
│   ├── sensitivity/     # accepted-size and rejected-size sensitivity analyses
│   ├── baseline/        # accepted-only baseline
│   └── diagnostics/     # isolated-applicant diagnostic
├── scripts/
│   └── make_thesis_figures.py            # recreates Figures 4.1–4.4 of the thesis
├── data/                # place the LendingClub files here (not included)
├── requirements.txt
└── LICENSE
```

## Data

The data come from the public LendingClub dataset on Kaggle:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

The raw files are not included because of their size. Download them and place them in `data/` as described in [data/README.md](data/README.md).

## Installation

Python 3.9 is recommended (the experiments were run with Python 3.9.18).

```bash
git clone https://github.com/peixocas/reject-inference-class-imbalance-label-spreading.git
cd reject-inference-class-imbalance-label-spreading
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

1. Place the data files in `data/`.
2. Open `notebooks/reject_inference_pipeline.ipynb` from the repository root (so that the relative paths `data/` and `results/` resolve correctly) and run the cells in order.

The notebook is organized in sections:

| Section | What it does | Output |
|---|---|---|
| Data import and Steps 1–2 | Feature harmonization and deterministic cleaning | — |
| Proposed Framework | Defines the pipeline (`run_one_seed`) | — |
| Main predictive evaluation | 20 seeds, N_L = 10,000, S = 3,000, γ = 2 | `results/main/` |
| Accepted-size / Rejected-size sensitivity | Change `N_L_value` or `S_value` in the cell and re-run for each configuration | `results/sensitivity/` |
| Label propagation behavior | γ sweep over 0.5, 1, 2, 5, 10 (10 seeds) | `results/gamma_sweep/` |
| Statistical tests | Paired Wilcoxon tests and Holm correction | — |
| Accepted-only baseline | Each pipeline vs. an identical twin without rejected applicants | `results/baseline/` |
| Isolated-applicant diagnostic | Rejected applicants that receive no label in the graph | `results/diagnostics/` |

To recreate the thesis figures from the saved results (no need to re-run the experiments):

```bash
python scripts/make_thesis_figures.py
```

### Computational notes

Label Spreading with a dense RBF kernel stores a similarity for every pair of nodes, so memory grows quadratically with graph size. The main configuration (about 16,000 nodes) runs on a standard machine, but the larger sensitivity configurations need substantially more memory (around 37 GB for N_L = 50,000). These were run on the NOVA Data & Analytics Hub (NOVA DAH).

### Reproducibility

All runs use fixed seeds (0–19, or 0–9 for the γ sweep) for data splits, sampling, oversampling, and models. Small numerical differences can still appear across computing environments and library versions, mainly for the MLP. The CSV files in `results/` are the exact outputs reported in the thesis.

## Citation

If you use this code, please cite the thesis:

> Bernardo, R. M. H. (2026). *Machine learning for credit risk modelling: A framework addressing reject inference and class imbalance* [Master's thesis, NOVA Information Management School, Universidade Nova de Lisboa].

```bibtex
@mastersthesis{bernardo2026rejectinference,
  author = {Bernardo, Rui Miguel Henriques},
  title  = {Machine Learning for Credit Risk Modelling: A Framework Addressing Reject Inference and Class Imbalance},
  school = {NOVA Information Management School, Universidade Nova de Lisboa},
  year   = {2026},
  type   = {Master's thesis}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The LendingClub data are subject to the terms of their source on Kaggle and are not redistributed here.

## Acknowledgements

I thank my supervisor, Professor Afshin Ashofteh, and NOVA Information Management School for access to the NOVA Data & Analytics Hub, which made the larger-sample experiments possible.
