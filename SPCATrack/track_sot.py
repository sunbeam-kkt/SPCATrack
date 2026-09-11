#!/usr/bin/env python3
"""Evaluate the revised manuscript's first-frame-initialized SOT protocol.

The search-region expansion factor is not specified in the manuscript,
so this script requires it explicitly instead of inventing a hidden default.
Use the exact factor from the reported experiment and state it in the manuscript
for full reproducibility.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from spcatrack.config import SPCATrackConfig
from spcatrack.context import build_context_encoder
from spcatrack.detector import UltralyticsDetector
from spcatrack.lgatracker import PairContextAdapter
from spcatrack.sot import SOTTracker, search_crop_from_box, offset_detections


def parse_box(text):
    vals=[float(x) for x in text.split(',')]
    if len(vals)!=4: raise argparse.ArgumentTypeError('box must be x,y,w,h')
    x,y,w,h=vals; return np.array([x,y,x+w,y+h],np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--first-box',required=True,type=parse_box,help='first-frame GT x,y,w,h; used once only')
    ap.add_argument('--weights',required=True)
    ap.add_argument('--assoc-checkpoint',required=True)
    ap.add_argument('--search-factor',required=True,type=float,help='must equal the factor used in the reported SOT experiment')
    ap.add_argument('--out',default='sot_result.txt')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--detector-device',default='0')
    args=ap.parse_args()

    cfg=SPCATrackConfig()
    detector=UltralyticsDetector(args.weights,args.detector_device,cfg)
    encoder=build_context_encoder('structured-mlp',args.device)
    cap=cv2.VideoCapture(args.video); ok,frame=cap.read()
    if not ok: raise RuntimeError('Cannot read video')
    h,w=frame.shape[:2]
    from spcatrack.lgatracker import LGATracker
    tracker=LGATracker.from_checkpoint((w,h),encoder,args.assoc_checkpoint,cfg,args.device)
    # Reuse loaded adapter in SOT specialization.
    sot=SOTTracker((w,h),encoder,tracker.adapter,cfg,args.device)
    tr=sot.initialize(args.first_box)
    rows=[(1,*args.first_box.tolist())]
    while True:
        ok,frame=cap.read()
        if not ok: break
        if tr is None: break
        crop,offset=search_crop_from_box(frame,tr.xyxy,args.search_factor)
        candidates=offset_detections(detector.detect(crop),offset)
        tr=sot.update_single(candidates)
        if tr is not None and tr.history[-1].frame_index == sot.frame_index:
            x1,y1,x2,y2=map(float,tr.xyxy); rows.append((sot.frame_index,x1,y1,x2-x1,y2-y1))
        else:
            rows.append((sot.frame_index,float('nan'),float('nan'),float('nan'),float('nan')))
    cap.release()
    np.savetxt(args.out,np.asarray(rows),fmt='%.6f',delimiter=',')
    print(f'saved: {args.out}')

if __name__=='__main__': main()
