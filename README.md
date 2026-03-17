# 🧠 EEG Neuroprosthetic Control – Performance Means Nothing Without Real-Time Viability

## 📌 Description

This project explores **motor intention decoding from raw EEG signals** for neuroprosthetic control, using the [Grasp-and-Lift EEG Detection dataset](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection) (Kaggle / Nature Scientific Data).

The central question is not just *"how accurate is the model?"* but **"is it actually usable in real life?"**

A model achieving AUC 0.895 may be clinically useless. A model achieving AUC 0.821 may change a patient's life. The difference? **Latency.**

---

## 🎯 Clinical Context

In neuroprosthetic control, a brain-machine interface must detect motor intentions in **real time**. Beyond ~300ms of total latency (window size + inference time), prosthetic control degrades significantly and the user's sense of embodiment is compromised.

This project benchmarks models on **two dimensions simultaneously**:
- Classification performance (AUC ROC)
- Real-time viability (total latency < 300ms)

This dual evaluation reflects the actual constraints of clinical BCI systems such as those developed by [NeuroRestore (EPFL)](https://www.neurorestore.epfl.ch/).

---

## 📊 Dataset

- **Source:** [Kaggle – Grasp-and-Lift EEG Detection](https://www.kaggle.com/competitions/grasp-and-lift-eeg-detection) — referenced in *Nature Scientific Data*
- **Subjects:** 12
- **Channels:** 32 EEG channels at 500 Hz
- **Task:** Detect 6 sequential motor events during object grasping and lifting
- **Class imbalance:** ~2.6% positive frames per event (~97.4% rest)
- **Total training data:** ~10 hours of EEG recordings

### The 6 Motor Events

| # | Event | Neuroprosthetic Relevance |
|---|-------|--------------------------|
| 1 | HandStart | Movement initiation |
| 2 | FirstDigitTouch | Object contact |
| 3 | BothStartLoadPhase | Load phase onset |
| 4 | LiftOff | Object lift |
| 5 | Replace | Object replacement |
| 6 | BothReleased | Final release |

> Labels are active within a ±150ms window around each event timestamp — defining the system's temporal precision requirement.

---

## ⚙️ Pipeline

```
Raw EEG (µV)
  → Notch filter (50 Hz)        # Remove power line noise
  → Low-pass filter (< 5 Hz)    # Preserve Bereitschaftspotential
  → Common Average Reference     # Remove common artifacts
  → Z-score normalization        # Equalize channel amplitudes
  → Sliding window features      # 720 features per window
  → LDA classification           # Per-event binary classifier
```

### Feature Extraction (720 features per window)

| Feature | Details | Count |
|---------|---------|-------|
| Band power | 5 frequency bands × 32 channels | 160 |
| Variance | Per channel | 32 |
| Covariance matrix | Upper triangle (32×32 SPD matrix) | 528 |

---

## 📈 Key Results

### Real-Time Benchmark — Subject 1

| Window | AUC (LDA) | Total Latency | Real-Time Viable? |
|--------|-----------|---------------|-------------------|
| 4s | 0.603 | 4001ms | ❌ |
| 1s | 0.895 | 1001ms | ❌ |
| **250ms** | **0.813** | **251ms** | **✅** |

> The 1s window achieves the best AUC — but is clinically unusable. The 250ms window is the only viable configuration.

### LOSO Validation — Subject 1

Leave-One-Series-Out (LOSO) validation was implemented to address temporal data leakage from the sequential, deterministic nature of the 6 motor events.

| Validation | AUC | Note |
|------------|-----|------|
| K-Fold (standard) | 0.788 | Potential temporal leakage |
| **LOSO (rigorous)** | **0.821** | Clinically defensible |

> Surprisingly, LOSO outperformed K-Fold (+0.034). K-Fold was mixing heterogeneous session contexts in train/validation splits, artificially degrading measured performance.

### Intra-Subject Generalization — All 12 Subjects

| Metric | Value |
|--------|-------|
| Mean AUC | 0.706 ± 0.049 |
| Min AUC | 0.632 (Subject 12) |
| Max AUC | 0.821 (Subject 1) |

**Strong inter-individual variability** was identified as the primary system limitation — not the algorithm or latency. Individual calibration is necessary but insufficient to guarantee uniform performance across subjects.

### Per-Event AUC — LOSO (Subject 1)

| Event | AUC | Note |
|-------|-----|------|
| HandStart | 0.908 🟢 | Strong Bereitschaftspotential — no sequential bias possible |
| BothStartLoadPhase | 0.888 🟢 | Clear mu/beta ERD |
| FirstDigitTouch | 0.866 🟢 | Somatosensory feedback |
| LiftOff | 0.811 🟡 | Documented M1 contralateral peak |
| BothReleased | 0.734 🟡 | Beta ERS — declining signal |
| Replace | 0.721 🟡 | Progressive deactivation |

> HandStart's high score (0.908) is the strongest evidence of genuine EEG decoding: as the *first* event in the sequence, its performance cannot be explained by sequential context learning.

---

## ⚠️ Methodological Limitations

1. **Model selection validated on Subject 1 only** — a rigorous comparison across all 12 subjects was computationally infeasible in this context
2. **Sequential protocol** — the deterministic event order (HandStart → ... → BothReleased) prevents full elimination of sequential bias, regardless of validation strategy
3. **Offline pipeline** — measured latency (251ms) is an estimate; embedded hardware introduces additional constraints
4. **Controlled conditions ≠ clinical reality** — neurotypical subjects, laboratory environment, no ecological artifacts

---

## 🔭 Next Steps

- **Inter-subject generalization** (Leave-One-Subject-Out) — computationally intensive, requires optimized feature caching
- **Riemannian Geometry** (pyRiemann) — state-of-the-art BCI approach operating directly on covariance matrices; naturally robust to inter-session signal drift
- **Adaptive calibration** — Bayesian LDA updated at session onset to compensate signal drift
- **EEGNet** — lightweight DL architecture specialized for EEG, potentially more robust on difficult sessions

```python
# Riemannian approach — next implementation
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

pipeline_riemannian = Pipeline([
    ('cov', Covariances(estimator='lwf')),
    ('ts',  TangentSpace(metric='riemann')),
    ('clf', LinearDiscriminantAnalysis())
])
```

---

## 🧾 Key Takeaways

- A high-AUC model with >300ms latency is **clinically unusable** for neuroprosthetic control
- LOSO validation is **mandatory** for time-series EEG data — standard K-Fold introduces temporal leakage
- Inter-individual variability, not algorithmic performance, is the **primary bottleneck** of current BCI systems
- HandStart decoding (AUC 0.908) provides strong evidence of genuine neural signal decoding rather than sequential pattern learning

---

## 📂 Repository Structure

```
eeg-neuroprosthetic-control/
├── notebook/
│   └── grasp_lift_eeg_detection.ipynb   # Full pipeline notebook
├── docs/
│   └── recap_eeg_neuro.docx             # Concepts reference document (FR)
├── results/
│   └── figures/                         # Generated visualizations
└── README.md
```

---

## 👤 Author

**Nicolas (9Nicox)**
PhD in Neuroscience (TMS, motor cortex plasticity) | 4 years neurorehabilitation research (Parkinson's, Stroke) | Data Scientist

This project is part of a portfolio targeting neurorehabilitation and brain-machine interface research positions in French-speaking Switzerland.

📅 2025
