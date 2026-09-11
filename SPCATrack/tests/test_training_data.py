import json
from pathlib import Path
import cv2
import numpy as np
import torch

from spcatrack.training.data import PairedMOTDataset, build_pair_manifest


def test_pair_manifest_and_collate(tmp_path: Path):
    frames = tmp_path / "frames" / "V1"
    anns = tmp_path / "ann"
    frames.mkdir(parents=True); anns.mkdir()
    im = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(frames / "000001.jpg"), im)
    cv2.imwrite(str(frames / "000002.jpg"), im)
    (anns / "V1.txt").write_text(
        "1,7,10,20,20,10,1,1,1.0\n2,7,12,20,20,10,1,1,1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "pairs.jsonl"
    assert build_pair_manifest(str(tmp_path / "frames"), str(anns), str(manifest)) == 1
    ds = PairedMOTDataset(str(manifest), imgsz=640, augment=False)
    item = ds[0]
    batch = PairedMOTDataset.collate_fn([item])
    assert batch["img"].shape == (2, 3, 640, 640)
    assert batch["bboxes"].shape == (2, 4)
    assert batch["track_ids"].tolist() == [7, 7]
    assert batch["pair_slot_per_image"].tolist() == [0, 1]
    assert batch["paper_pair_mode"] is True
