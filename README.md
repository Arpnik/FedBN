## Extended Evaluation & Findings (This Repository)

This repository builds upon the original work **FedBN: Federated Learning on Non-IID Features via Local Batch Normalization**
by Li et al. (ICLR 2021).  
All original ideas, formulations, and baseline contributions of FedBN are fully credited to the original authors.

The goal of this extension is **not to propose a new federated algorithm**, but to **critically evaluate the robustness and
practical limitations of FedBN** under more realistic federated learning conditions that were not explored in the original
paper.

---

### Motivation

The original FedBN work demonstrates strong performance under *feature-shift non-IID settings*, assuming:
- One domain per client
- Clean local datasets
- Domain-aligned Batch Normalization statistics

In real-world federated systems, these assumptions are often violated. Clients may:
- Collect data from **multiple domains over time**
- Contain **asymmetric noise** (e.g., label noise, sensor noise)
- Operate under mixed and evolving data distributions

This repository extends the original experiments to examine **when FedBN’s advantages persist—and when they degrade**.

---

### Summary of Findings

Our experiments first reproduce the original FedBN results and then systematically extend them across three dimensions.

#### 1. Reproduction of Original FedBN Results

We faithfully reproduce the core results reported in the FedBN paper on the Office-Caltech benchmark, confirming that:
- FedBN outperforms FedAvg and FedProx under clean, single-domain-per-client settings
- Client-specific Batch Normalization effectively mitigates feature shift

This validates the correctness of the implementation and establishes a reliable baseline.

---

#### 2. Robustness Under Asymmetric Client-Side Noise

We introduce heterogeneous noise across clients, including:
- Additive Gaussian input noise
- Label noise with varying intensities

**Key observation:**  
FedBN remains robust when each client corresponds to a single domain, outperforming FedAvg and FedProx on most seen
domains. This indicates that client-specific normalization can stabilize feature distributions even under noisy supervision.

---

#### 3. Mixed-Domain Clients (Violation of One-Domain-Per-Client Assumption)

We evaluate FedBN in settings where individual clients contain data from **multiple domains**, violating a core assumption
of the original method.

**Key observation:**  
FedBN’s advantage consistently degrades under mixed-domain clients. Batch Normalization statistics become entangled
across heterogeneous feature distributions, weakening FedBN’s ability to isolate domain-specific shifts. In these settings,
performance differences between FedBN, FedAvg, and FedProx become marginal or unstable.

This highlights a critical dependency of FedBN on **domain-aligned client partitioning**.

---

#### 4. Normalization Layer Ablation

We further analyze whether FedBN’s gains generalize beyond Batch Normalization by replacing BN with:
- Group Normalization
- Layer Normalization

**Key observation:**  
FedBN’s performance gains are intrinsically tied to Batch Normalization. When BN is replaced, FedBN’s advantage largely
disappears, confirming that its effectiveness relies on preserving client-specific batch statistics rather than normalization
in general.

---

### Key Takeaways

- FedBN is highly effective under **clean, single-domain client settings**
- It remains robust to **asymmetric noise** when domain boundaries are preserved
- Its advantages **break down under mixed-domain clients**
- Performance gains do **not generalize to non-batch-based normalization methods**

These findings clarify both the **strengths and limitations** of FedBN and motivate future work on:
- Domain-aware client modeling
- Robust normalization strategies
- Federated methods that explicitly handle intra-client heterogeneity

---

### Detailed Report

A complete description of the experimental setup, datasets, noise configurations, ablation studies, and results is provided
in the accompanying report included in this repository:

📄 **[final-report.pdf](./final-report.pdf)**

Readers interested in experimental details and quantitative analysis are encouraged to consult the report.

---
### Future Directions

While this work provides a systematic evaluation of FedBN under more realistic federated settings, several limitations
remain and motivate future research:

- **Limited client heterogeneity modeling**: Mixed-domain clients are constructed using fixed domain mixtures. Future work
  could study *dynamic domain drift*, where client data distributions evolve across communication rounds.


- **Noise–domain interaction not exhaustively explored**: Although noise and domain mixing are evaluated independently,
  their combined effect is not fully characterized. Further experiments could analyze compounding instability arising from
  simultaneous noise and domain shift.


- **Focus on vision benchmarks**: Experiments are limited to standard image-based domain adaptation datasets. Extending
  this analysis to other modalities (e.g., text, speech, medical time-series) would test the generality of the findings.


- **No algorithmic modification proposed**: This work focuses on evaluation rather than proposing a new method. Future
  research could leverage these findings to design hybrid or adaptive normalization schemes that relax the one-domain-per-
  client assumption.

---

### Attribution

This work is an **extension and empirical evaluation** of the original FedBN method.  
All core algorithmic ideas and original contributions belong to:

> Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou  
> *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization*  
> ICLR 2021

This repository aims to complement the original work by exploring **practical robustness and failure modes** under
realistic federated learning settings.
