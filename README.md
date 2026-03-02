# 🌐 Fed-Medicaid: SignGuard
**ECDSA-Based Byzantine Defense for Federated Healthcare Fraud Detection**

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-70%2F70_passed-brightgreen.svg)]()

Machine learning is a critical tool for identifying anomalous provider behavior in national healthcare systems. However, centralizing highly sensitive claims data (like Medicare/Medicaid) poses severe compliance and privacy risks (HIPAA). **Federated Learning (FL)** offers a paradigm to collaboratively train fraud detection models across decentralized institutional silos (US states) without transferring raw patient histories.

However, Vanilla FL is wildly vulnerable to **Byzantine failures** and **Data Poisoning** via Sybil networks. This repository contains the implementation of **SignGuard**, a hybrid cryptographic defense mechanism evaluated against the largest public healthcare dataset to date—227 million real-world HHS Medicaid provider claims partitioned across 54 United States jurisdictions.

## 🔐 The SignGuard Architecture

SignGuard strictly decouples identity non-repudiation from gradient validation.

```text
[State A] ──(Gradient + ECDSA Sig)─► [  Federation Server  ] ◄─(Gradient + ECDSA Sig)── [State B]
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
             [ Cryptographic Gate ]                 [ Statistical Validator ]
             Verify(PublicKey, Sig)                 1. L2-Norm Bounding Check
             Reject Unauthorized                    2. Cosine Similarity Vector
                          │                                     │
                          ├─────────────────────────────────────┘
                          ▼
                 [ Reputation Ledger ]
             Exponential Moving Average (EMA)
             Update Client Score R_i ∈ [0, 1]
                          │
                   [ Aggregation ]
```

### 1. Cryptographic Identity (ECDSA NIST P-256)
All participants generate an Elliptic Curve key pair. The aggregation server drops any update failing verification, neutralizing network spoofing.

### 2. Statistical Validation & Reputation
Accepted identities have their parameter gradients mathematically audited:
* **L2 Norm Thresholding:** Reject structural explosions inherent to gradient ascent poisoning.
* **Cosine Alignment:** Angle determination against the historical global memory vector to detect coordinated subversion.
* **Reputation Ledger (EMA):** Rejections immediately decay trust exponentially, permanently isolating Sybil participants.

## 📊 Empirical Evaluation Matrix
The repository includes automated execution wrappers validating the resilience against:
- **Random Poisoning** (Sensory failures)
- **Model Poisoning** (Gradient Ascent / Targeted degradation)
- **Label Flipping** (Collusion data-poisoning)
- **Free Riding** (Compute theft)
- **Sybil Networks** (Coordinated identity spoofing)

SignGuard actively neutralizes these structural and Sybil manipulation injections while preserving AUPRC within **1.8%** of theoretically optimal centralized baselines.

## 🚀 Quick Start & Reproduction

### Prerequisites
1. Ensure the P1 `medicaid-guard` feature generation pipeline has been executed on the primary dataset.
2. The `fed-medicaid` system will ingest these partitions automatically.

### Installation
```bash
git clone https://github.com/alazkiyai09/fed-medicaid.git
cd fed-medicaid
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Experiments
Execute the built-in experimental wrappers orchestrating the FL loop across 54 state boundaries:
```bash
# E1: Centralized vs. Federated Baseline
python experiments/run_experiment_1.py

# E2: Defense comparison (Clean environment benchmarking)
python experiments/run_experiment_2.py

# E3: Attack robustness 5x5 Matrix (Attacks vs Defenses)
python experiments/run_experiment_3.py

# E4: Differential Privacy trade-off sweep (Local Gaussian Mechanism)
python experiments/run_experiment_4.py

# E5: System Scalability across client participation permutations
python experiments/run_experiment_5.py
```

## 📝 Research & Citations
For the full mathematical formulations, performance extrapolations, and differential privacy equations governing SignGuard's behavior under the Local Gaussian Mechanism, refer to the included `/paper/arxiv_draft.md` specification.

## ⚖️ License
This project is open-sourced under the MIT License. Data used is public domain US government property.
