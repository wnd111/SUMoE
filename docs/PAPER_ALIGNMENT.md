# Paper-to-code alignment

This release treats the displayed equations and experimental specification in
the Finley manuscript as the normative description. The table below records
the executable location of each component so that a reviewer can audit the
implementation without inferring an unreported mechanism.

| Paper item | Executable implementation | Alignment invariant |
|---|---|---|
| Candidate forests | `sumoe/forest/parsers/`, `calibration.py`, `builder.py` | Stanza contributes one candidate; spaCy and Stack-Transformer use beam width 16 and return up to five. Parser-local temperature probabilities are multiplied by LAS-derived priors, duplicate labeled trees are merged by mass, and the highest-mass five are renormalized. |
| Eqs. (1)-(3) | `sumoe/forest/types.py`, `alignment.py`, `collate.py` | Candidate posteriors sum to one; edge marginals are posterior sums; sentence-local forests are mapped into document token space with no cross-sentence dependency edge. |
| Eq. (4) | `sumoe/model/forest_encoder.py` | `alpha_ij = softmax_j(A_ij / tau)` over every valid source position. A missing sparse edge has `A_ij = 0` and contributes `exp(0)`; padding and target positions are excluded. There is no learned query/key term. |
| Eq. (5) | `sumoe/model/forest_encoder.py` | `FFN(W_V h_j)` is computed before the all-source weighted sum. The implicit zero-edge baseline plus sparse corrections is algebraically equal to the displayed dense formula. |
| Eq. (6) | `sumoe/model/forest_encoder.py` | A sigmoid gate with input width `2d` fuses the structural vector and decoder state after configured decoder blocks 4, 8, 12, 16, 20, 24, 28, and 32; the eight injection modules have independent parameters. |
| Eq. (7) | `sumoe/model/tree_readout.py`, `router.py` | Two four-head GAT layers produce root-plus-mean tree readouts of dimension 256; candidate readouts are posterior-weighted, sentence summaries are averaged, and masked semantic max pooling is projected into the 1024-dimensional routing space. |
| Eq. (8) | `sumoe/model/router.py` | The document routing vector is linearly projected and softmax-normalized over eight experts. |
| Eq. (9) | `sumoe/model/router.py`, `experts.py` | Deterministic Top-2 selection evaluates only selected document-expert pairs. Retained full-softmax probabilities are not renormalized after selection. |
| Expert block | `sumoe/model/experts.py` | Each independent expert contains pre-RMSNorm eight-head causal self-attention and a residual RMSNorm-SwiGLU sublayer with intermediate width 4096 and dropout 0.1. |
| Eq. (10) | `sumoe/model/losses.py`, `training/trainer.py` | Utilization is the hard Top-2 assignment frequency over the effective global batch and is normalized by `B*k`; `lambda_bal = 0.05` is applied exactly once. The straight-through estimator changes the gradient path, not the displayed loss value. |
| Eq. (11) | `sumoe/model/forest_encoder.py`, `sumoe_model.py` | Cosine alignment is computed only for valid source-token dependent positions with retained forest edges and is averaged over injection layers. |
| Eq. (12) | `sumoe/model/sumoe_model.py` | Total loss is task loss plus the already weighted balance term plus `lambda_forest * forest loss`, with `lambda_forest = 1.0`. |
| Task losses | `sumoe/data/tasks.py`, `sumoe_model.py`, `losses.py` | Generation uses target-only causal cross-entropy. QuALITY and ContractNLI use four-class and three-class cross-entropy, respectively. No fixed task-to-expert assignment is imposed. |

## Main experimental contract

The canonical `configs/model/sumoe.yaml` fixes the manuscript configuration:
LLaMA-3.1-8B-Instruct, 4096 total tokens, 512 generated tokens, eight independent
forest injections after decoder blocks 4, 8, 12, 16, 20, 24, 28, and 32, five
forest candidates, eight experts, Top-2 routing,
global batch 64, AdamW learning rate `3e-5`, betas `(0.9, 0.98)`, weight decay
`0.01`, three epochs, and seeds 13, 21, and 42. Deterministic evaluation uses
beam size one with sampling disabled.

These invariants are guarded by `tests/test_paper_release_alignment.py` and the
component-level numerical tests. The configuration and code make the reported
experimental procedure executable; they do not substitute for the original
datasets, parser models, learned checkpoints, predictions, or run logs.
