# SPCATrack paper-code alignment

This source tree is aligned to the manuscript titled **SPCATrack: Semantic
Pseudo-Segmentation and Context-Guided Association for UAV Tracking**.

## Stage I: ISPS-SOD

| Manuscript component | Implementation |
|---|---|
| YOLOv13-S, one UAV class | `configs/yolov13s-UAV.yaml` |
| Adjacent-frame identity pairs | `spcatrack/training/data.py` |
| SPS: `F -> A -> M_F, M_B` | `spcatrack/sps.py::SPSBranch` |
| Box-conditioned `q_sep` diagnostic | `spcatrack/sps.py::q_sep` |
| Foreground/background contrastive loss | `foreground_background_contrastive_loss` |
| MSA-NWD | `spcatrack/training/model.py::ScaleAttentionNWD` |
| TCR | `PaperAlignedDetectionModel._tcr` |
| MIFD | `mifd_loss` |
| BNS-HM | `bns_hard_negative_loss` |
| Complete weighted detector objective | `PaperAlignedDetectionModel.loss` |
| 50 epochs, batch 16, 640, AdamW, AMP | `train.py` |
| Table-1 component and 1280-pixel controls | `train.py --detector-ablation/--imgsz` |

SPS is applied to the high-resolution P3 feature stream. Detector-positive
locations used by the auxiliary objectives are recomputed from the same native
YOLOv13 task-aligned assignment used by the decoupled detection loss. TCR uses
only identities valid in both adjacent frames, while MIFD and TCR use the shared
SPS response restricted by individual ground-truth boxes.

## Stage II: LGATracker

| Manuscript component | Implementation |
|---|---|
| Five window attributes (QS, SS, AP, MD, AC) | `summarize_window`, `WindowSummary.prompt` |
| Task-specific fixed tokenizer | `StructuredPromptTokenizer` |
| Trainable `E_tok`, `d_t=64` | `StructuredPromptEncoder.token_embedding` |
| Mean pooling + GELU MLP 64->128->128 | `StructuredPromptEncoder` |
| `W_g: 4->128`, `W_e: 128->128` | `PairContextAdapter` |
| Position, size, state, direction cues | `LGATracker.pairwise_cues` |
| Cue weights .50/.20/.10/.20 | `SPCATrackConfig.cue_weights` |
| `W_s=30`, `K=10`, `L_max=5` | `SPCATrackConfig` |
| Causal completed-window history | `LGATracker._complete_window` |
| Context compatibility and `beta=.20` fusion | `PairContextAdapter.compatibility`, `_score_matrix` |
| Hungarian cost `1-S` | `LGATracker.update` |
| Joint mapper/projection BCE training | `scripts/train_association.py` |
| Geometry-only association control | `context_variant=geometry-only` |
| Table-4 cue/attribute removals | `--disable-cue`, `--disable-attribute` |

The proposed method has no pretrained language backbone. The detector is frozen
during Stage-II training; `E_tok`, the language MLP, `W_g`, and `W_e` are
optimized jointly. Stage-II defaults follow the stated 50-epoch, batch-16,
AdamW (`lr=5e-4`, weight decay `0.01`), cosine-decay, five-epoch-warm-up
protocol. Hungarian assignment remains inference-only.

## Controlled semantic-content study

`controlled_prompt` and `scripts/train_association.py --variant` implement:

- `constant`;
- `shuffled-window`, drawing only from earlier completed windows with a fixed seed;
- `attribute-permuted`, independently drawing QS, SS, AP, MD, and AC from earlier completed windows;
- `numeric`, the NV + MLP control;
- `full`, the proposed structured prompt.

All controls reuse `W_s=30`, `K=10`, the 128-D context space, the same pair
features, and the same association loss. The first shuffled/permuted window has
no prior causal pool and uses the neutral constant template.

## Naming cleanup

The executable package is only `spcatrack`. Legacy package names, duplicate
configurations, outdated architecture artwork, and pretrained-language paths
are absent so that an earlier implementation cannot be imported by mistake.

## Reproducibility boundary

Source-level alignment does not by itself reproduce the numerical tables. Exact
reproduction also requires the authors' dataset split, detector-to-GT matching
exports, trained checkpoints, three random-seed logs, and A100 environment.
These private experiment assets are not fabricated in this archive.
