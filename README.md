# SPCATrack

Official implementation of **SPCATrack: Semantic Pseudo-Segmentation and
Context-Guided Association for UAV Tracking**.

SPCATrack is a two-stage framework:

1. **ISPS-SOD** augments a YOLOv13-S detector with semantic
   pseudo-segmentation (SPS), multi-scale attention normalized Wasserstein
   distance (MSA-NWD), temporal consistency regularization (TCR),
   multi-instance feature decorrelation (MIFD), and background negative sample
   hard mining (BNS-HM).
2. **LGATracker** combines four detection-track cues with strictly causal,
   completed-window context and performs Hungarian assignment.

The proposed language path uses a fixed task-specific tokenizer, a trainable
64-D token embedding, non-padding mean pooling, and a two-layer GELU MLP with
dimensions `64 -> 128 -> 128`. It does **not** use CLIP, MiniLM, an LLM, or
another pretrained language encoder.

## Repository layout

```text
SPCATrack-main/
├── configs/
├── spcatrack/
│   ├── context.py
│   ├── lgatracker.py
│   ├── sps.py
│   └── training/
├── scripts/
│   ├── build_pair_manifest.py
│   ├── build_assoc_pairs_from_mot.py
│   ├── train_association.py
│   ├── benchmark_e2e.py
│   └── benchmark_peak_memory.py
├── train.py
├── track_mot.py
├── track_sot.py
└── tests/
```

See `PAPER_CODE_ALIGNMENT.md` for the equation/module correspondence.

## Environment

The manuscript experiments use Python 3.11, PyTorch 2.2.2, CUDA 12.1, and an
NVIDIA A100. Create the environment and install the official YOLOv13 dependency:

```bash
conda env create -f environment.yml
conda activate spcatrack
bash scripts/bootstrap_yolov13.sh
```

The detector architecture is the official iMoonLab YOLOv13-S graph with
`nc=1`; details are recorded in `upstream/YOLOV13_PIN.md`.

## Stage I: train ISPS-SOD

Build adjacent-frame pairs from Track-3 frame folders and MOT-format identity
annotations:

```bash
python scripts/build_pair_manifest.py \
  --frames-root /path/to/Track3/frames \
  --annotations-root /path/to/Track3/Annotations \
  --out data/track3_train_pairs.jsonl
```

Expected annotation rows are:

```text
frame,id,x,y,w,h,conf,class,visibility
```

Train the complete detector objective:

```bash
python train.py \
  --pair-manifest data/track3_train_pairs.jsonl \
  --model configs/yolov13s-UAV.yaml \
  --data configs/MOT-UAV.yaml \
  --device 0
```

The defaults reproduce the reported protocol: 50 epochs, image batch size 16
(eight adjacent-frame pairs), 640x640 input, AdamW, initial learning rate
`5e-4`, weight decay `0.01`, cosine decay, five warm-up epochs, and AMP. The
native YOLOv13 detection loss retains unit weight. Auxiliary weights are:

| Term | Weight |
|---|---:|
| foreground/background contrastive loss | 0.10 |
| MSA-NWD | 0.50 |
| TCR | 0.10 |
| MIFD | 0.05 |
| BNS-HM | 0.10 |

Additional constants are `tau_con=0.10`, `eta_nwd=0.50`, MIFD
`alpha=0.30`, `tau_soft=0.10`, and BNS-HM `r=0.02`, `gamma_h=0.30`.
`q_sep` is logged to `paper_aux_metrics.csv` as a diagnostic only; it is not
part of the objective or checkpoint-selection rule.

The detector switch matrix and resolution control in the manuscript are
reproduced without editing source code:

```bash
python train.py ... --detector-ablation w/o-tcr --imgsz 640
python train.py ... --detector-ablation w/o-sps --imgsz 640
python train.py ... --detector-ablation full --imgsz 1280
```

Available rows are `full`, `base`, `w/o-tcr`, `w/o-mifd`, `w/o-stcr`,
`w/o-bns-hm`, `w/o-msa-nwd`, and dependency-aware `w/o-sps`. In the last row,
TCR, MIFD, and BNS-HM are also disabled while MSA-NWD remains enabled, matching
the paper table.

## Stage II: train LGATracker

The detector is frozen. First build a causal pair/context corpus from detector
outputs matched to ground-truth identities:

```bash
python scripts/build_assoc_pairs_from_mot.py \
  --mot train_detections_with_gt_id.txt \
  --width 640 \
  --height 512 \
  --out data/assoc_pairs.jsonl
```

Jointly train the task-specific token embedding, language MLP, `W_g`, and `W_e`:

```bash
python scripts/train_association.py \
  --corpus data/assoc_pairs.jsonl \
  --variant full \
  --out weights/lgatracker.pt \
  --seed 0
```

Stage II uses the same declared optimization protocol: 50 epochs, batch size
16, AdamW with initial learning rate `5e-4` and weight decay `0.01`, followed
by cosine decay after a five-epoch linear warm-up.

The dimensions match the manuscript:

```text
E_tok: V x 64
f_mlp: 64 -> 128 -> 128 (GELU)
W_g:   4 -> 128
W_e:   128 -> 128
```

Hungarian assignment is used only for online inference and is not
differentiated through.

## Online multi-UAV tracking

```bash
python track_mot.py \
  --weights runs/spcatrack/isps_sod/weights/best.pt \
  --assoc-checkpoint weights/lgatracker.pt \
  --input dataset/Videos \
  --output processed_results \
  --context-variant full \
  --detector-device 0 \
  --device cuda
```

The reported configuration uses `W_s=30`, history `K=10`, `L_max=5`,
`d_s=128`, and `beta=0.20`. Pairwise cue weights are
`(0.50, 0.20, 0.10, 0.20)` for position, size, state, and direction. Scale
states use original-frame box areas: small `<16^2`, large `>96^2`, and medium
otherwise. Motion uses the original-frame diagonal `D=sqrt(W^2+H^2)` and
threshold `50/D`.

For frames in window `r`, only embeddings from completed windows through
`r-1` are available. The current window is encoded after its final association
and affects only the next window.

## Controlled semantic-content experiments

The Table-8 variants use the same mapper capacity, window length, training
schedule, and association protocol:

```bash
for variant in full constant shuffled-window attribute-permuted numeric; do
  python scripts/train_association.py \
    --corpus data/assoc_pairs.jsonl \
    --variant "$variant" \
    --out "weights/lgatracker_${variant}.pt" \
    --seed 0
done
```

- `constant`: the same neutral template is used for every completed window.
- `shuffled-window`: each window draws a prompt, with a fixed seed, only from
  prompts completed earlier in the same sequence.
- `attribute-permuted`: QS, SS, AP, MD, and AC are independently drawn from
  earlier completed windows in the same sequence.
- `numeric`: the 10-D numeric window vector is mapped directly by an MLP.

Use the same `--context-variant` and seed for training and evaluation. The first
window of the shuffled/permuted controls has no earlier causal pool and uses the
neutral template.

The detector-controlled Geometry-only row requires no association checkpoint:

```bash
python track_mot.py \
  --weights weights/isps_sod_best.pt \
  --input dataset/Videos \
  --output processed_results_geometry \
  --context-variant geometry-only
```

The Table-4 single-component removals use the same Stage-II command and
checkpoint/runtime flags. For example:

```bash
python scripts/train_association.py \
  --corpus data/assoc_pairs.jsonl \
  --variant full \
  --disable-cue pos \
  --out weights/lgatracker_without_pos.pt

python track_mot.py \
  --weights weights/isps_sod_best.pt \
  --assoc-checkpoint weights/lgatracker_without_pos.pt \
  --input dataset/Videos \
  --disable-cue pos
```

Replace `pos` with `size`, `state`, or `dir`; use `--disable-attribute` with
`qs`, `ss`, `ap`, `md`, or `ac` for the five prompt-attribute rows. Checkpoints
store these switches and inference refuses a mismatched configuration.

## Single-object tracking protocol

`track_sot.py` uses the ground-truth box only in the first frame, then evaluates
class-agnostic candidates with the same pairwise and context-aware score. It
does not use later ground truth for correction or re-initialization:

```bash
python track_sot.py \
  --video sequence.mp4 \
  --first-box x,y,w,h \
  --weights weights/isps_sod_best.pt \
  --assoc-checkpoint weights/lgatracker.pt \
  --search-factor <verified_experiment_value>
```

The manuscript does not state a numerical search-region expansion factor, so
the script requires the verified experimental value rather than hiding an
invented default.

## Efficiency protocol

The end-to-end benchmark uses batch size 1, 640x640 detector input, AMP, 100
warm-up frames, 1,000 timed frames, and CUDA synchronization around every timed
iteration. It includes preprocessing, detector forward inference, pairwise
cues, context fusion, Hungarian assignment, track update, and the amortized
once-per-window prompt mapper; file I/O, serialization, and visualization are
excluded.

```bash
python scripts/benchmark_e2e.py \
  --video dataset/Videos/MultiUAV-002.mp4 \
  --weights weights/isps_sod_best.pt \
  --assoc-checkpoint weights/lgatracker.pt
```

## Tests

```bash
python test.py
```

The tests cover detector-loss shapes, temporal pairing, causal context, the
semantic controls, exact language dimensions, GELU activation, cue weights,
state thresholds, and diagonal-normalized motion.

## Reproducibility boundary

The source aligns formulas, data flow, dimensions, constants, and protocols
with the manuscript. Numerical AP/MOTA/FPS reproduction additionally requires
the exact Track-3 split, trained checkpoints, seed-specific logs, and A100
runtime used for the reported experiments; those assets are not included.
