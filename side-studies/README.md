# Cross-Modal Validation Studies

> Two complementary side studies on tabular behavioural datasets that triangulate the demographic findings from the main AFHN facial-image work.

## Why these are here

The main AFHN work (in [`../notebook/`](../notebook/)) audits demographic bias on a facial-image dataset and finds substantial subgroup performance gaps. A natural question is whether these patterns reflect a genuine clinical signal (age and sex really matter for ASD detection) or are artifacts of the specific image dataset.

To triangulate this, we ran two side studies on **tabular behavioural/clinical datasets** using a deliberately different modeling approach (**gradient-boosted trees** via XGBoost, with SHAP for interpretation). The reasoning: if subgroup patterns are consistent across image-based deep learning *and* tabular gradient boosting on completely different datasets, the patterns are more likely to reflect real clinical structure than dataset artifacts.

These studies are **exploratory triangulation**, not primary evidence. They are documented thoroughly so anyone reproducing the main work can see the broader picture.

---

## Study 1: Children & Adolescents (ages 1–18)

**Notebook:** [`children_tabular_xgboost.ipynb`](children_tabular_xgboost.ipynb)
**Dataset:** [Kaggle: ASD Screening for Children and Adolescents](https://www.kaggle.com/datasets/uppulurimadhuri/dataset) (~2,000 records)
**Features:** 10 behavioural questionnaire items (A1–A10), demographics, clinical comorbidities (depression, anxiety, genetic disorders, speech delay, intellectual disability)

### Key results

| Metric | Value |
|---|---|
| Overall accuracy | **87.9%** |
| Girls (pooled model) | **99.1%** |
| Boys (pooled model) | **83.7%** |
| Girls-only model | 97.2% |
| Boys-only model | 84.8% |

### What SHAP revealed

Sex was the **top-ranked feature** in SHAP, ahead of all 10 behavioural questionnaire items. Depression and family history of ASD were also strong predictors.

![SHAP summary for children](figures/children_shap_summary.png)

### Connection to AFHN

The Children dataset shows girls are *easier* to classify, while AFHN's facial-image work shows a modest male-favoured gap. The directional inconsistency is itself informative — suggesting the AFHN gap may be driven by facial-feature distribution rather than the questionnaire-detectable behavioural-presentation differences known to make ASD harder to detect in girls clinically.

---

## Study 2: Toddlers (ages 12–36 months)

**Notebook:** [`toddlers_tabular_xgboost.ipynb`](toddlers_tabular_xgboost.ipynb)
**Dataset:** [Kaggle: Early Autism Screening Dataset for Toddlers](https://www.kaggle.com/datasets/ajithdari/early-autism-screening-dataset-for-toddlers) (~1,000 records)
**Features:** Behavioural scores (social interaction, communication, repetitive behaviour, language delay), demographics, jaundice history, family ASD history

### Key results

| Metric | Value |
|---|---|
| Overall accuracy | **46%** (below chance for binary classification) |

### Why the low accuracy is actually the finding

Without SHAP analysis, 46% accuracy looks like a failure. With SHAP, it becomes a clinically-meaningful result: the model's confidence stratifies by developmental phase in a way that mirrors documented paediatric milestones.

![Developmental phase analysis](figures/toddler_developmental_phases.png)

- **12–18 months (Ambiguous Phase):** SHAP impact is *negative* (median ~−0.2). The model recognises this as a phase where neurotypical and autistic toddlers exhibit largely overlapping behaviours, and (correctly) refuses to commit to confident predictions.
- **19–24 months (Divergence Phase):** SHAP impact is near zero. Diagnostic signals are beginning to emerge but are not yet stable.
- **25–36 months (Complex Social Phase):** SHAP impact spreads widely (−1.0 to +1.5). The model becomes confident, and is correct more often than not.

**The interpretation:** the model's low aggregate accuracy is not a modeling failure but a *faithful reflection of a clinical reality* — early ASD screening before ~24 months is genuinely difficult because the diagnostic signal has not yet emerged.

### Connection to AFHN

Both AFHN's age-bin analysis and the Toddler study show predictability concentrating in older subjects. The Toddler study's SHAP-grounded interpretation provides a candidate mechanism: ASD-relevant signatures simply have not yet diverged enough from neurotypical baselines in very young children. For AFHN, this implies the model is likely most reliable for older children — and any deployment should explicitly age-stratify reported accuracies rather than report a single aggregate number.

---

## How to run these notebooks

Both notebooks are designed to run in **Google Colab** (the datasets are loaded via the Kaggle API).

1. Open the notebook in Colab.
2. Get a Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account) → "Create New API Token" → upload `kaggle.json` to the Colab session.
3. Run all cells.

Each notebook is self-contained and takes ~5 minutes to run.

## Files in this folder

| File | Description |
|---|---|
| `children_tabular_xgboost.ipynb` | Study 1 notebook |
| `toddlers_tabular_xgboost.ipynb` | Study 2 notebook |
| `figures/children_shap_summary.png` | SHAP feature-importance plot for the Children dataset |
| `figures/toddler_developmental_phases.png` | Developmental-phase analysis for the Toddler dataset |
| `results/children_subgroup_accuracy.csv` | Per-sex accuracy breakdown for Study 1 |
| `results/toddlers_overall_accuracy.csv` | Headline numbers for Study 2 |
| `results/toddlers_developmental_phase.csv` | Phase-by-phase SHAP interpretation |

## Limitations

These are exploratory triangulation studies, not primary contributions:

- **No rigorous cross-validation** — single 80/20 stratified split per dataset
- **No threshold tuning** — used XGBoost defaults
- **No demographic audit beyond age and sex** — race wasn't in either dataset
- **Different ground-truth labelling protocols** across the two datasets, making direct cross-dataset comparison imperfect
- **Self-reported / parent-reported behavioural items** — known to be noisy

The findings are meant to triangulate the AFHN bias story, not replace it. For the rigorous treatment, see the main notebook and the report in [`../report/`](../report/).
