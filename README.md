# SSRL · Anonymous core implementation

Core implementation accompanying **Semantic-guided Stable Representation Learning for Incremental Unified Multimodal Anomaly Detection**.

The package contains the semantic anchors, shared Transformer encoder and independent modality decoders, state-space cross-modal interaction, geometric reliability refinement, reconstruction/topology/drift losses, and the frozen-reference incremental update. It accepts aligned RGB and geometric features produced externally by a frozen feature backbone and MFCN.

## Files

- `ssrl/model.py`: core representation reconstruction and refinement.
- `ssrl/losses.py`: the manuscript's reconstruction, topology and drift objectives.
- `ssrl/continual.py`: previous-stage initialization, frozen reference and one training update.
- `ssrl/__init__.py`: package exports.

Experiment configurations, YAML/JSON config files, dataset manifests, training schedules, dataset preprocessing, evaluation utilities, model weights and experiment logs are not included. This is a core-code release for inspection, not an end-to-end reproduction package. Model constructor parameters are explicit in the implementation; there is no external experiment configuration.

## Interface

The core model accepts `feature_align` and `feature_align_xyz`, each shaped [B,C,H,W]. For the manuscript feature interface C=272 and H=W=14. An optional `image` tensor determines the upsampled anomaly-map resolution. The text source is frozen CLIP ViT-B/32. CLIP text embeddings may instead be supplied directly for offline inspection or tests. The reference selective scan backend runs on CPU; CUDA execution uses a compatible mamba-ssm installation.

Supporting imports require Python 3.8+, PyTorch and the CLIP dependencies. There is no automatic training command and no bundled configuration or data. `prepare_stage` accepts a previous core-model state dictionary, initializes the current model and produces a frozen reference. At the base stage it returns no reference. `train_step` updates only the current model using the same current-stage inputs for both models.

## Validation

The supplied manuscript was used to correct the shared encoder, independent decoder positions, final-representation reconstruction loss and raw-token topology graph. Tests cover tensor dimensions, exact objective reductions, refinement and anomaly maps, gradients, deterministic inference, analytic dropout variance, frozen-reference updates and state-dictionary round trips. Synthetic tests do not validate the paper's reported experimental results. Full GPU training was not rerun.

The old two-encoder state dictionary is incompatible with this corrected structure and cannot be loaded directly. Full experiment configurations remain local.

For review, use the anonymous endpoint provided with the submission. No author identity, personal link or original Git history is included in this artifact. The landing page includes a noindex directive; the actual review endpoint must also be checked before submission.
