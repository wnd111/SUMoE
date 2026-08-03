# Public release scope

This archive is a source-only public release. It is designed to make the Finley
manuscript's method and experimental configuration auditable and rerunnable
without redistributing third-party or large learned artifacts.

Included:

- Python source for forest construction, SUMoE, baselines, training, inference,
  evaluation, profiling, and statistical analysis;
- canonical and ablation configuration files;
- unit and integration tests;
- public implementation documentation and MIT license.

Not included:

- LLaMA, Stanza, spaCy, Stack-Transformer, or other model weights;
- SUMoE, baseline, parser, DCC, optimizer, or scheduler checkpoints;
- SCROLLS, Penn Treebank, generated dependency forests, or other datasets;
- predictions, run manifests, TensorBoard/W&B data, metric reports, or profiler
  outputs;
- manuscript table values represented as newly generated machine outputs.

Consequently, this archive supports independent reproduction after users obtain
the required data and licensed model resources and run the documented pipeline.
It does not claim that the manuscript's numerical results were regenerated from
files contained in the archive. The release builder enforces this boundary by
excluding data, checkpoint, output, report, cache, and model-artifact paths and
common weight/archive suffixes.
