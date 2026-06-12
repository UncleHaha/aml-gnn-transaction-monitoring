# Transaction Monitoring for Financial Crime Detection

### Explainable Graph Neural Networks for Anti-Money Laundering: A Systematic Comparison of Architectures and Class-Imbalance Strategies

This repository contains the full experimental framework for a Master's research project at the Faculty of Business and Economics, University of Malaya. The project provides a systematic, multi-dataset, multi-seed comparison of Graph Neural Network (GNN) architectures and class-imbalance treatments for anti-money laundering (AML) transaction monitoring.

---

## Overview

Money laundering is a relational phenomenon enacted through networks of accounts and intermediaries. Graph Neural Networks model transactions as graphs and capture the multi-hop, network-level patterns that rule-based and tabular models miss. This study benchmarks three GNN architectures against three class-imbalance treatments across two public datasets, evaluating each configuration over five random seeds.

The full experimental sweep is the Cartesian product of:

| Factor | Options | Count |
| --- | --- | --- |
| Architectures | GCN, GAT, GraphSAGE | 3 |
| Imbalance strategies | Weighted CE, Focal Loss, GraphSMOTE | 3 |
| Datasets | Elliptic, IT-AML (HI-Small) | 2 |
| Random seeds | 0, 1, 2, 3, 4 | 5 |
| **Total** | | **90** |

Of the 90 planned runs, 85 completed successfully; the five IT-AML x GAT x GraphSMOTE runs failed with CUDA out-of-memory errors (documented as a limitation).

## Key Findings

- **GraphSAGE + Focal Loss** is the strongest configuration on Elliptic (AUC-PR 0.465), a 51.7% relative improvement over the next-best configuration.
- **GCN + GraphSMOTE** leads on IT-AML by the operationally relevant AUC-PR (0.089, a 24x improvement over the random-classifier baseline at 0.37% test-set illicit rate).
- The optimal class-imbalance treatment depends on imbalance severity: focal loss suffices under moderate imbalance, while data-level oversampling (GraphSMOTE) becomes necessary in the long-tail regime characteristic of real AML.
- Cross-seed variance is substantial for GCN on both datasets, underscoring the need for multi-seed reporting.

## Repository Structure

```
.
├── configs/                 # Experiment configuration matrix (90 runs)
├── scripts/
│   ├── run_experiments.py   # Main experimental sweep
│   └── aggregate_results.py # Mean +/- std aggregation across seeds
├── src/
│   ├── data/                # Dataset loaders (Elliptic, IT-AML)
│   ├── models/              # GNN architectures (GCN, GAT, GraphSAGE)
│   ├── losses/              # Weighted CE, Focal Loss
│   ├── sampling/            # GraphSMOTE oversampling
│   └── training/            # Training loop and metrics
├── results/                 # Per-run JSON results and aggregated summary CSV
├── requirements.txt         # Python dependencies
└── AML_GNN_Notebook_FINAL.ipynb  # End-to-end notebook (Colab-compatible)
```

## Setup

Requires Python 3.11+.

```bash
# Clone the repository
git clone https://github.com/UncleHaha/aml-gnn-transaction-monitoring.git
cd aml-gnn-transaction-monitoring

# Install dependencies
pip install -r requirements.txt
```

## Datasets

- **Elliptic Bitcoin dataset** -- automatically downloaded by PyTorch Geometric on first run.
- **IBM IT-AML (HI-Small)** -- must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and placed in `data/it_aml/` (both `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`).

The `data/` directory is excluded from version control due to file size.

## Running Experiments

Run the full 90-configuration sweep:

```bash
python scripts/run_experiments.py
```

Run a single configuration:

```bash
python scripts/run_experiments.py --dataset elliptic --architecture graphsage --imbalance_strategy focal --seed 0
```

Aggregate results across seeds:

```bash
python scripts/aggregate_results.py
```

## Methods

The experimental design and metric definitions correspond to the accompanying research article. The principal algorithms are documented as pseudocode in the article's Appendix A and implemented as follows:

| Component | Implementation |
| --- | --- |
| Elliptic loader | `src/data/elliptic_loader.py` |
| IT-AML loader | `src/data/itaml_loader.py` |
| GNN architectures | `src/models/gnn_models.py` |
| Focal loss | `src/losses/losses.py` |
| GraphSMOTE | `src/sampling/graphsmote.py` |
| Training loop | `src/training/trainer.py` |
| Metrics (AUC-PR, Recall@k) | `src/training/metrics.py` |
| Experimental sweep | `scripts/run_experiments.py` |

Evaluation uses AUC-PR as the primary metric (appropriate under extreme class imbalance), with F1, Recall@k, and AUC-ROC reported alongside.

## Citation

If you use this code, please cite the accompanying research article:

```
Goh, E. S. X. (2026). Transaction Monitoring for Financial Crime Detection.
Master's research project, Faculty of Business and Economics, University of Malaya.
```

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

This work uses the Elliptic Bitcoin dataset (Weber et al., 2019) and the IBM IT-AML synthetic dataset (Altman et al., 2023). Implementation builds on PyTorch and PyTorch Geometric.
