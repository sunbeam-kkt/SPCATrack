#!/usr/bin/env python3
"""Build causal Stage-II supervision from identity-labelled MOT detections.

The output is streaming JSONL rather than precomputed language features because the
manuscript jointly trains the task-specific token embedding, the two-layer MLP,
and W_g/W_e. Summary records are written once per completed window; pair
records store only the last K causal summaries, avoiding quadratic duplication
of the full sequence history.

Expected MOT rows (comma or whitespace separated):
    frame, identity, x, y, w, h, confidence, ...
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from spcatrack.config import SPCATrackConfig
from spcatrack.lgatracker import LGATracker, PairContextAdapter, coarse_state, summarize_window
from spcatrack.types import Detection, Track, TrackObservation


class _UnusedEncoder:
    output_dim = 128

    def encode(self, prompt, numeric=None):  # pragma: no cover - never called here
        raise RuntimeError("The corpus builder must not encode context.")


def load_mot(path: str) -> dict[int, list[tuple[int, Detection]]]:
    frames: dict[int, list[tuple[int, Detection]]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [part for part in re.split(r"[\s,]+", line.strip()) if part]
        if len(parts) < 6:
            continue
        frame, identity = int(float(parts[0])), int(float(parts[1]))
        x, y, width, height = map(float, parts[2:6])
        if width <= 0 or height <= 0:
            continue
        confidence = float(parts[6]) if len(parts) > 6 else 1.0
        box = np.asarray([x, y, x + width, y + height], dtype=np.float32)
        frames[frame].append((identity, Detection(box, confidence)))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot", required=True, help="detections matched to GT identities")
    parser.add_argument("--width", type=int, required=True, help="original video-frame width")
    parser.add_argument("--height", type=int, required=True, help="original video-frame height")
    parser.add_argument("--out", required=True, help="output JSONL corpus")
    parser.add_argument("--neg-per-pos", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = SPCATrackConfig()
    helper = LGATracker(
        (args.width, args.height),
        _UnusedEncoder(),
        PairContextAdapter(cfg.context_dim, cfg.shared_dim),
        cfg,
        device="cpu",
    )
    frames = load_mot(args.mot)
    if not frames:
        raise RuntimeError("No valid MOT detections were found.")

    active: dict[int, Track] = {}
    last_seen: dict[int, int] = {}
    window_observations: list[TrackObservation] = []
    history = deque(maxlen=cfg.context_history)
    completed = []
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    sequence = Path(args.mot).stem
    pair_count = 0
    positives = 0
    with output.open("w", encoding="utf-8") as stream:
        def write_record(record: dict) -> None:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        for frame_index in range(1, max(frames) + 1):
            current = frames.get(frame_index, [])
            for identity in list(active):
                if frame_index - last_seen.get(identity, frame_index) > cfg.max_lost:
                    active.pop(identity, None)
                    last_seen.pop(identity, None)

            # Context is strictly causal: only completed previous windows are stored.
            if history and active:
                history_indices = list(range(len(completed) - len(history), len(completed)))
                history_records = [summary.to_dict() for summary in history]
                for identity, detection in current:
                    detection.state = coarse_state(detection.area, cfg)
                    candidates = [
                        (other_id, helper.pairwise_cues(detection, track))
                        for other_id, track in active.items()
                    ]
                    positives_for_detection = [candidate for candidate in candidates if candidate[0] == identity]
                    negatives = [candidate for candidate in candidates if candidate[0] != identity]
                    chosen = positives_for_detection[:1]
                    if negatives:
                        count = min(len(negatives), args.neg_per_pos if positives_for_detection else 1)
                        indices = rng.choice(len(negatives), size=count, replace=False)
                        chosen.extend(negatives[int(index)] for index in np.atleast_1d(indices))
                    for other_id, cues in chosen:
                        label = 1.0 if other_id == identity else 0.0
                        write_record(
                            {
                                "record_type": "pair",
                                "sequence": sequence,
                                "frame_index": frame_index,
                                "window_index": (frame_index - 1) // cfg.window_size + 1,
                                "pair_features": cues.tolist(),
                                "history_indices": history_indices,
                                "history_summaries": history_records,
                                "label": label,
                            }
                        )
                        pair_count += 1
                        positives += int(label == 1.0)

            # GT identities update the training-only tracks after pair construction.
            for identity, detection in current:
                detection.state = coarse_state(detection.area, cfg)
                if identity not in active:
                    active[identity] = Track(
                        identity,
                        detection.xyxy.copy(),
                        detection.confidence,
                        detection.state,
                    )
                track = active[identity]
                track.xyxy = detection.xyxy.copy()
                track.confidence = detection.confidence
                track.state = detection.state
                track.lost = 0
                observation = TrackObservation(
                    frame_index,
                    identity,
                    detection.center.copy(),
                    detection.xyxy.copy(),
                    detection.confidence,
                    detection.state,
                )
                track.history.append(observation)
                window_observations.append(observation)
                last_seen[identity] = frame_index

            if frame_index % cfg.window_size == 0:
                summary = summarize_window(
                    window_observations,
                    args.width,
                    args.height,
                    cfg.stationary_pixels,
                )
                summary_index = len(completed)
                completed.append(summary)
                history.append(summary)
                write_record(
                    {
                        "record_type": "summary",
                        "sequence": sequence,
                        "summary_index": summary_index,
                        "summary": summary.to_dict(),
                    }
                )
                window_observations = []

    if not pair_count:
        output.unlink(missing_ok=True)
        raise RuntimeError("No pairs were produced; at least two windows and recurring identities are required.")
    print(f"saved {pair_count} causal pairs -> {output}; positives={positives}")


if __name__ == "__main__":
    main()
