# MCI-Classification

Training and analysis framework for **"An Integrated Learning Framework for White Matter
Connectivity-Based Cognitive Impairment Staging: Backbone-Specific Enhancement and
Clinical Phenotype Interpretation"**.

Whole-brain 90×90 white matter structural connectivity matrices (AAL atlas) are used for
three-class staging (NC / EMCI / LMCI) with four backbone networks — Transformer, ResNet,
FCN and LSTM — each paired with an architecture-specific enhancement module (CBAM, ASPP,
SE, SGE; Supplementary Methods S1–S4). The framework reproduces the full paper protocol:
two-stage Optuna Bayesian tuning with 10-fold CV, 100 repeated random hold-out
experiments with paired baseline/enhanced splits, one-vs-rest ACC/AUC/SEN/SPE metrics,
edge-level feature importance with stage-specific top-20 connections, Mantel tests with
FDR correction, Pearson correlation, and Lasso + SHAP clinical-score interpretation
(MMSE / FAQ / GDS).

## License

This repository is licensed under a **Review and Academic Non-Commercial Use License**
(see [LICENSE](LICENSE)). Briefly: the code may be used for peer review of the manuscript
and for non-commercial academic verification of its results. Commercial use, clinical
use, application to other datasets, and any use unrelated to the manuscript require prior
written permission from the corresponding author.

## Data

**No data are included in this repository.** The ADNI-derived connectivity matrices and
clinical scores must be obtained under the researcher's own ADNI data use agreement
(http://adni.loni.usc.edu). Expected layout:

```
data/
├── subjects.csv                 # subject_id, group(NC/EMCI/LMCI), mmse, faq, gds [, age, sex, education]
├── connectivity/{subject_id}.npy  # 90x90 white matter structural connectivity
└── md/{subject_id}.npy            # 90x90 mean diffusivity (used by the analysis stage)
```

A deterministic mock cohort (NC=54, EMCI=42, LMCI=54) can be generated to exercise the
entire pipeline before real data are available:

```bash
python scripts/make_mock_data.py
python scripts/validate_data.py
```

## Environment

The paper environment (Section 2.6) is Ubuntu 22.04, CUDA 12.1, Python 3.10.12,
PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121, 2 × RTX 4090 (24 GB):

```bash
conda env create -f environment.yml && conda activate mci
# or: pip install -r requirements.txt
```

## Pipeline

Every stage is a plain script with a matching VSCode debug configuration
(`.vscode/launch.json`); all scripts accept `--fast` for reduced-scale debugging.

```bash
# 1. tune one variant (two-stage Optuna, 20+10 trials, 10-fold CV)
python scripts/run_tuning.py --config configs/transformer_cbam.yaml

# 2. 100 repeated hold-out runs; baseline+enhanced share identical splits
python scripts/run_holdout.py --configs configs/transformer.yaml configs/transformer_cbam.yaml

# 3. aggregate runs into Table 2 / metric distributions / ROC data / top-20 connections
python scripts/summarize.py

# 4. clinical association and explainability (Mantel+FDR, Pearson, Lasso+SHAP)
python scripts/run_analysis.py

# two-GPU orchestration of all 8 variants (tune + hold-out), one job per GPU
python scripts/schedule_all.py --stage all
```

Interrupted hold-out sweeps resume automatically (existing `runs/run_XXX.json` are
skipped); `--shard 0/2` / `--shard 1/2` splits the 100 seeds across two GPUs.

## Outputs

```
outputs/
├── <variant>/runs/run_XXX.json     # split indices, hyperparameters, epoch history,
│                                   # OvR metrics, softmax probabilities, edge importance
├── <variant>/tuning/               # best_params.json, trials.csv
└── summaries/
    ├── table2.csv / table2.md      # paper Table 2 layout (Base vs Enhanced, mean ± std)
    ├── metrics_distribution.csv    # Fig 5 data
    ├── roc_data.npz                # Fig 6 data
    ├── top20_EMCI.csv, top20_LMCI.csv, shared_vs_specific.csv   # Fig 7/8 data
    ├── mantel_results.csv          # Fig 4 data (r, permutation p, FDR q)
    ├── corr_matrix_*.csv           # Pearson correlation between connections
    └── shap/<stage>_<score>/       # Fig 9/10 data (SHAP values, top-12 contributions)
```

## Repository layout

```
configs/        default.yaml + one config per model variant
src/data/       datasets, stratified splits, mock-data generator
src/models/     Transformer/ResNet/FCN/LSTM backbones + CBAM/ASPP/SE/SGE modules
src/training/   Trainer (AdamW, warm-up+cosine, early stopping, AMP), Optuna tuner,
                100-run hold-out engine
src/evaluation/ one-vs-rest metrics, permutation / gradient edge importance
src/analysis/   Mantel tests, Pearson correlation, Lasso + SHAP, result tables
scripts/        entry points (training, tuning, hold-out, scheduling, summary, analysis)
```

## Citation

If you use this code for permitted academic purposes, please cite the manuscript.

## Contact

Permission requests and questions: Jun Wang `<wangjun_haust@163.com>`.
