"""Single-object tracking protocol described in the revised manuscript.

Only the first-frame GT box is used. Later frames use the previous target
estimate to define a search region, generate class-agnostic detector candidates,
and choose the candidate with the highest SPCATrack fused association score.
No later-frame GT correction or re-initialization is permitted.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np

from .config import SPCATrackConfig
from .types import Detection, Track, TrackObservation
from .lgatracker import LGATracker, coarse_state


def search_crop_from_box(frame: np.ndarray, xyxy: np.ndarray, factor: float):
    if factor <= 1.0:
        raise ValueError("search_factor must be > 1.0 and must match the value used in the reported SOT experiment.")
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = map(float,xyxy)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    bw,bh=max(2.0,x2-x1),max(2.0,y2-y1)
    sw,sh=bw*factor,bh*factor
    rx1=max(0,int(round(cx-sw/2))); ry1=max(0,int(round(cy-sh/2)))
    rx2=min(w,int(round(cx+sw/2))); ry2=min(h,int(round(cy+sh/2)))
    if rx2 <= rx1 or ry2 <= ry1:
        return frame, (0,0)
    return frame[ry1:ry2,rx1:rx2], (rx1,ry1)


def offset_detections(detections: Sequence[Detection], offset: Tuple[int,int]):
    ox,oy=offset
    out=[]
    for d in detections:
        b=d.xyxy.copy(); b[[0,2]] += ox; b[[1,3]] += oy
        out.append(Detection(b,d.confidence,d.cls,d.state))
    return out


class SOTTracker(LGATracker):
    """One-active-track specialization of LGATracker."""
    def initialize(self, gt_xyxy: np.ndarray, confidence: float = 1.0):
        if self.frame_index != 0:
            raise RuntimeError("SOT initialization is allowed only in the first frame.")
        self.frame_index = 1
        d=Detection(np.asarray(gt_xyxy,dtype=np.float32),confidence,0)
        d.state=coarse_state(d.area,self.cfg)
        self._new_track(d)
        if self.frame_index % self.cfg.window_size == 0:
            self._complete_window()
        return next(iter(self.tracks.values()))

    def update_single(self, candidates: Sequence[Detection]) -> Optional[Track]:
        if not self.tracks:
            self.frame_index += 1
            if self.frame_index % self.cfg.window_size == 0:
                self._complete_window()
            return None
        self.frame_index += 1
        tr=next(iter(self.tracks.values()))
        dets=[self._prepare_detection(d) for d in candidates]
        if dets:
            scores=self._score_matrix(dets,[tr])[:,0]
            i=int(np.argmax(scores))
            if float(scores[i]) >= self.cfg.min_match_score:
                self._update_track(tr,dets[i])
            else:
                tr.lost += 1
        else:
            tr.lost += 1
        if tr.lost > self.cfg.max_lost:
            self.tracks.clear()
            tr=None
        if self.frame_index % self.cfg.window_size == 0:
            self._complete_window()
        return tr
