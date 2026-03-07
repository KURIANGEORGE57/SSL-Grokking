# Label-JEPA: Cross-Modal Latent Alignment on Modular Arithmetic

**The first counterexample (to our knowledge) to the Fourier circuit consensus in grokking research.**

A JEPA architecture, trained to align latent representations of input pairs $(a, b)$ with learned encodings of the target $c = (a+b) \bmod p$, groks modular addition — reaching 100% validation accuracy after a prolonged memorization phase — **without developing the Fourier circuits that every prior study identified as the mechanism of grokking on this task.**

---

## What Is This?

Every previous study of grokking on modular arithmetic (Nanda et al., 2023; Zhong et al., 2023; Power et al., 2022; Liu et al., 2022) found the same result: the model embeds inputs onto a circle and uses discrete Fourier transforms with trigonometric identities to perform addition as rotation. This became the de facto theory — grokking on modular addition *means* Fourier circuit formation.

This repository provides the first counterexample. Same task ($p = 97$, 30/70 split), same generalization outcome (100% validation accuracy), **flat Fourier spectrum throughout training**. The difference: a cosine similarity loss in learned latent space (JEPA) instead of cross-entropy classification.

### The Core Finding

**The loss function determines the generalization mechanism.** Same data, same label access, same architectural backbone — swap cross-entropy for latent alignment and Fourier circuits disappear while grokking persists. This demonstrates that Fourier circuits are a property of the cross-entropy pathway, not a property of the task itself.

---

## How It Works

### Architecture: Label-JEPA

The model uses a standard JEPA architecture with three components:

- **Context encoder**: Embeds input pair $(a, b)$ via shared embeddings → MLP → L2 normalization
- **Predictor**: Maps context latents to predicted target latents through a bottleneck (128 → 64 → 64 → 128)
- **Target encoder (EMA)**: Embeds the target residue $c$ via embedding → MLP → L2 normalization. Updated via exponential moving average, not gradient descent.

The loss is negative cosine similarity between the predictor output and the target encoder's representation. **No classification head. No cross-entropy. No logits over classes.**

### What Kind of Learning Is This?

The model never predicts $c$ directly. It predicts a *learned, evolving latent representation* of $c$ — one that changes every step via EMA. From the model's perspective, this is indistinguishable from predicting a latent of an augmented view.

This is structurally analogous to **CLIP**: an image encoder takes the image, a text encoder takes the caption (which is a human annotation), and they are trained to align in latent space via cosine similarity. CLIP is universally considered self‑supervised (or sometimes “weakly supervised,” but never **supervised classification**). Label-JEPA follows the same logic — the label enters through a learned encoder, not through a classification objective.

| | CLIP | Label-JEPA |
|---|---|---|
| Input encoder receives | Image | $(a, b)$ |
| Target encoder receives | Human-written caption | $c = (a+b) \bmod p$ |
| Target is a human annotation? | Yes | Yes |
| Loss function | Cosine similarity in latent space | Cosine similarity in latent space |
| Model predicts labels directly? | No | No |

The label $c$ plays two distinct roles that prior work conflates:

1. **Task specification** — it tells the model *which* operation is being learned (addition, not multiplication or XOR). This role is irreducible for algebraic tasks (see Theoretical Arguments below).
2. **Mechanism selection** — *how* the model internally represents the solution is determined not by the label itself, but by the loss function. This is the novel finding.

---

## Key Results

### 1. Non-Fourier Grokking

The target encoder maintains a flat Fourier spectrum throughout training (top-5 concentration ≈ 0.071, near the uniform baseline of 0.052). The context encoder develops mild spectral concentration during memorization (~0.39) that *decreases* during grokking (~0.21). Generalization coincides with the **dismantling** of partial Fourier structure, not its formation.

### 2. Reversed Geometric Dynamics

During supervised grokking, effective rank typically decreases (representations compress). In Label-JEPA grokking, effective rank **increases** during the generalization phase — the model expands its representational dimensionality while generalizing.

### 3. Training Loss Is Blind to Grokking

The JEPA loss barely changes during the generalization transition (≈ −0.91 → −0.96 while validation accuracy goes from ~10% to 100%). The memorizing and generalizing solutions occupy nearly the same loss basin.

### 4. Near-Orthogonal Codebook

The target encoder discovers a near-orthogonal set of codes for the 97 residue classes. The representation is high‑dimensional and distributed, contrasting with the low‑rank circular embedding found in supervised models.

---

## Theoretical Arguments

### Why pure SSL cannot grok algebraic tasks

Given the pair $(3, 4)$, what is the answer? The question is undefined. The pair belongs to class 7 under addition, class 12 under multiplication, class 96 under subtraction, class 25 under $(a^2 + b^2) \bmod p$, and infinitely many other operations. The input does not specify which operation is being learned.

This is a logical impossibility, not an empirical limitation. No architecture, no augmentation strategy, no amount of compute can make $(3, 4)$ intrinsically mean "addition" rather than "multiplication." The equivalence class structure does not exist in the input — it exists only in the choice of operation.

**Consequence**: For algebraic tasks where multiple operations share the same input space, the label is not a convenience — it is the task specification itself. This is why Label-JEPA requires $c$, and why removing it (as in standard SSL) makes the problem undefined.

### Why strong‑inductive‑bias SSL is label‑free only in form

One could design a "sum‑preserving augmentation" that maps $(a, b) \to (a+k, b-k) \bmod p$ — formally SSL since no label appears. But designing this augmentation requires knowing *in advance* that addition is the relevant operation. The human has encoded the answer into the augmentation. Change the task to multiplication, and you must redesign the entire augmentation scheme.

Label-JEPA is honest about where the task specification lives: in the data. Change from addition to multiplication by changing one line of data generation. The architecture, loss, and training procedure remain identical. The model is operation‑agnostic; the data specifies the operation.

---

## Repository Structure
├── self-supervised-algorithmic-generalization using JEPA.ipynb # Core notebook (15 sections)
├── extended-grokking.ipynb # Prediction accuracy & memorization dynamics
├── extended-grokking-2.ipynb # Target encoder intervention & predictor ablation
├── extended-grokking-3.ipynb # Fourier decomposition, causal intervention, task variation
└── LICENSE # Apache 2.0

text

### Notebook Details

| Notebook | Experiments | Establishes |
|----------|------------|-------------|
| `self-supervised-algorithmic-generalization using JEPA.ipynb` | Full pipeline (Sections 1–15) | Grokking demonstration, Fourier analysis, geometric dynamics, mechanistic probing (additive decomposition, predictor linearity, commutativity, linear decodability), supervised baseline, multi-seed validation |
| `extended-grokking.ipynb` | 1, 2, 8 | Direct JEPA prediction accuracy on unseen pairs; cosine similarity train/val divergence; linear probe vs direct prediction timeline |
| `extended-grokking-2.ipynb` | 3, 3b, 4 | Target encoder freeze at different training phases; EMA drift rate analysis; predictor ablation (identity replacement) |
| `extended-grokking-3.ipynb` | 5, 6, 7 | Fourier decomposition of class-mean latents with incremental reconstruction; causal band ablation & aligned vs orthogonal noise injection; task variation (multiplication, XOR, polynomial) |

---

## Setup & Reproduction

### Requirements
torch
numpy
matplotlib
scikit-learn

text

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Modulus (p) | 97 | Prime, standard for grokking |
| Train fraction | 0.3 | 30/70 split |
| Latent dim | 128 | Representation space |
| Hidden dim | 256 | Encoder hidden layer |
| Predictor bottleneck | 64 | Information bottleneck |
| Learning rate | 1e-3 | AdamW |
| Weight decay | 1.0 | High (standard for grokking) |
| EMA decay | 0.996 | Target encoder momentum |
| Epochs | 100,000 | Grokking onset varies by seed (20k–60k) |

### Running

Open any notebook in Jupyter and run all cells. The core notebook is self-contained and includes multi-seed validation (seeds 7, 123, 2024) and a supervised baseline for direct comparison. Extended notebooks build on the core run for intervention experiments.

All experiments are reproducible with `SEED = 42`.

---

## What This Is Not

This work does **not** claim:

- ~~"Fourier circuits are wrong"~~ — Nanda et al. correctly described what transformers with cross-entropy learn. This work shows it is one pathway, not the only one.
- ~~"A new architecture"~~ — the architecture is standard JEPA. The novelty is the data routing (input, label) and what it reveals about the role of the loss function.

This work **does** claim:

- **First counterexample (to our knowledge)** to the Fourier circuit consensus for modular addition grokking.
- **The loss function determines the generalization mechanism** — same task, same label access, different internal algorithm.
- **Task specification and mechanism selection are separable** — the label determines *what* is learned, the loss determines *how* it is internally represented.

---

## Open Questions

- What is the learned algorithm if not Fourier circuits? The mechanistic analysis reveals near-orthogonal codes with a Fourier expansion in the sum, but the complete picture — especially how the MLP computes this expansion — remains open.
- Does the non-Fourier pathway extend to all modular operations? The task variation notebook (Exp 7) provides initial evidence for multiplication, XOR, and polynomials, but further investigation is needed.
- What is the precise role of each component (EMA, bottleneck, weight decay, cosine loss) in producing the non-Fourier solution? Ablating any one collapses the system, but their interaction is not yet formally characterized.
- How does architecture (MLP vs transformer) interact with the loss function in determining the generalization mechanism? Your initial transformer experiments hint at rich interactions (2H works, 4H fails) that deserve deeper study.

---

## References

- Power, A. et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* arXiv:2201.02177
- Nanda, N. et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023
- Zhong, Z. et al. (2023). *The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks.* NeurIPS 2023
- Liu, Z. et al. (2022). *Towards Understanding Grokking: An Effective Theory of Representation Learning.* NeurIPS 2022
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence.* OpenReview
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

*Research and implementation by Kurian George.*
