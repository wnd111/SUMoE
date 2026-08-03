# SUMoE Implementation Notes

This document maps the paper's SUMoE components to the executable implementation.

## Forest-weighted aggregation

The executable implementation follows the displayed Eqs. (4) and (5)
literally. For every source token pair, including pairs absent from the sparse
edge list, Eq. (4) uses:

```text
alpha_ij = exp(A_ij / tau) / sum_{j'=1}^T exp(A_ij' / tau)
```

An absent edge has `A_ij = 0` and therefore contributes `exp(0) = 1` to the
denominator; it is not converted to a masked `-inf` logit. No learned token
compatibility term is added to Eq. (4). Padding, prompt scaffolding, and target
positions are outside the document source sequence and are excluded from `T`.
The implementation evaluates this dense mathematical result with an equivalent
sparse-edge correction formula, avoiding an explicit `T x T` allocation.

Following Eq. (5), the value projection and two-layer feed-forward
transformation are applied to every source position before the all-token
weighted sum:

```text
tilde_h_i = sum_j alpha_ij * FFN(W_V h_j)
```

The result is fused with the decoder state by the sigmoid gate in Eq. (6).
Eight independently parameterized Eq. (4)-(6) forest-injection modules are
applied after LLaMA decoder blocks 4, 8, 12, 16, 20, 24, 28, and 32. No second
attention equation or sparsified mask is applied on top of the displayed
equations.

The Eq. (11) cosine-alignment loss is evaluated only at valid source-token
dependent positions that have at least one retained forest edge. Fusion still
applies to every valid source position, as required by Eqs. (4)-(6).

## Experts

Each expert is a lightweight causal transformer block with the same hidden
dimension as the backbone. It contains pre-norm multi-head self-attention,
followed by a pre-norm SwiGLU feed-forward sublayer, dropout, and residual
connections. Only the selected document-expert pairs are evaluated. The MEO
condition imports the trained SUMoE backbone, semantic router, task heads, and
expert blocks; it then forms one document-conditioned transformer expert by
merging every expert-block parameter and evaluates that merged block once. It
is intentionally rejected by the independent-training CLI.

For SUMoE, the selected Eq. (9) coefficients remain the corresponding entries
of the full expert softmax; they are not renormalized over Top-2. The task-loss
straight-through path uses the hard sparse forward value and exactly one soft
probability gradient.

During cached autoregressive decoding, the implementation retains the
pre-expert backbone states from the prompt and prior generated tokens. The
selected causal expert block is reevaluated over that history and the current
token, while the backbone continues to use its ordinary key/value cache. This
keeps cached decoding numerically consistent with a full-prefix expert forward
pass.

## Multitask heads and task loss

The unified multitask sampler shuffles the available training examples without
family oversampling or a fixed task-to-expert mapping. GovReport,
SummScreenFD, QMSum, Qasper, and NarrativeQA use the shared autoregressive
language-model head. QuALITY uses a four-class head and maps the official
answer text to/from its `(A)`--`(D)` option block. ContractNLI uses a
three-class head and emits the official `Not mentioned` spelling. A two-logit
token span head is available as a model-level API, but none of these seven
SCROLLS tasks supplies span supervision. The task loss selects the applicable head;
prompt, source, and padding positions remain masked for generation loss.

## Parameter accounting

The executable model is the source of truth for parameter counts. With the
official LLaMA-3.1-8B configuration, the eight-injection implementation has
9,648,166,417 parameters: 8,030,261,248 backbone, 671,186,944 forest
injections, 939,720,704 experts, 6,960,648 tree-readout/router, and 36,873
task-head parameters. The revised paper does not state a conflicting fixed
total. The count command derives these values from the configured model.

This architecture changes model and optimizer state structure. Earlier
equation-alignment and pre-alignment checkpoints are intentionally rejected and
cannot be resumed; strict reproduction requires training a new checkpoint.

## Load-balancing loss

The implementation follows Eq. (10) of the revised paper. For a batch of `B`
documents and top-`k` routing, expert utilization is the hard selection
frequency divided by `B * k`. A straight-through estimator preserves this
hard Top-k value in the forward pass while propagating router gradients through
the soft probabilities. Eq. (10) does not define a gradient estimator for its
discrete selections. This implementation therefore fixes the estimator scope
explicitly: training accumulates stop-gradient document-feature snapshots over
all local micro-batches in one optimizer step, re-routes them together, and
all-reduces hard usage across workers. The auxiliary balance gradient updates
router parameters only; it does not propagate through the snapshots into the
backbone or tree readout. Task and forest losses continue to train those
upstream modules normally. This avoids retaining eight complete 4,096-token
backbone graphs solely for one discrete auxiliary statistic. Hence `B=64` is
the effective global batch rather than the physical per-device batch of one.
The reported balance term already includes both the
`1 / N_e` normalization and `lambda_bal`; it is added to the task loss exactly
once. Eq. (11) is implemented as task loss plus the already weighted balance
term plus the forest cosine-alignment loss; no additional `0.01` multiplier is
applied to the forest term.

## Profiling semantics

The profiler exposes separate `prefill` and `generation` modes. Prefill reports
parser, forest construction, model-forward, and complete prefill-pipeline
latencies; these are the metrics intended for table-style efficiency reporting.
Generation separately reports one-token time to first token, cached deterministic
greedy generation latency, and parse-through-completed-generation latency. Its
throughput uses the number of tokens actually produced, including early EOS.

## Distribution policy

Source code and configuration files are distributed. Model weights, training
checkpoints, datasets, parser resources, generated forests, predictions, and
reports are not distributed. Users must obtain third-party dependencies under
their respective access terms and licenses.
