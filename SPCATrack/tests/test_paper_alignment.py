import math
import numpy as np
import torch

from spcatrack.config import SPCATrackConfig
from spcatrack.context import StructuredPromptEncoder, StructuredPromptTokenizer
from spcatrack.types import Detection, TrackObservation
from spcatrack.sps import SPSBranch, mifd_loss, temporal_consistency_loss, bns_hard_negative_loss
from spcatrack.lgatracker import (
    LGATracker,
    PairContextAdapter,
    WindowSummary,
    coarse_state,
    controlled_prompt,
    summarize_window,
)


class DummyEncoder(torch.nn.Module):
    output_dim = 128
    def encode(self, prompt, numeric=None):
        # deterministic non-zero context for testing causal update
        return torch.ones(128)


def obs(frame, tid, x, y, conf=0.8, state=0):
    return TrackObservation(frame, tid, np.array([x,y], np.float32), np.array([x-2,y-2,x+2,y+2], np.float32), conf, state)


def test_state_thresholds():
    cfg = SPCATrackConfig()
    assert coarse_state(255, cfg) == 0
    assert coarse_state(256, cfg) == 2
    assert coarse_state(9217, cfg) == 1


def test_sps_dimensions():
    x = torch.randn(2, 16, 8, 10)
    a, mf, mb = SPSBranch(16)(x)
    assert a.shape == (2,1,8,10)
    assert mf.shape == x.shape and mb.shape == x.shape
    assert torch.allclose(mf + mb, x, atol=1e-5)


def test_motion_diagonal_normalization():
    # D=100 for 60x80; 60 px horizontal motion > 50 px threshold.
    o = [obs(1,1,0,0), obs(2,1,60,0)]
    s = summarize_window(o, 80, 60, 50.0)
    assert abs(s.delta_x - 0.6) < 1e-6
    assert s.motion == "horizontal"


def test_tcr_and_mifd():
    p0 = torch.tensor([[1.,0.],[0.,1.]])
    p1 = torch.tensor([[1.,0.],[1.,0.]])
    ids = torch.tensor([1,2])
    tcr = temporal_consistency_loss(p0, ids, p1, ids)
    assert tcr > 0
    assert temporal_consistency_loss(p0, ids, p0, ids).abs() < 1e-7
    same = torch.tensor([[1.,0.],[1.,0.]])
    assert mifd_loss([same], alpha=0.3, tau_soft=0.1) > 0


def test_causal_window_context():
    cfg = SPCATrackConfig(window_size=2)
    adapter = PairContextAdapter(128,128)
    tr = LGATracker((640,480), DummyEncoder(), adapter, cfg, device="cpu")
    d1 = Detection(np.array([10,10,20,20],np.float32),.9)
    tr.update([d1])
    assert tr.causal_context is None
    d2 = Detection(np.array([12,10,22,20],np.float32),.9)
    tr.update([d2])
    # Context appears only after frame 2 association is complete.
    assert tr.causal_context is not None
    assert len(tr.completed_summaries) == 1


def test_pair_descriptor_order_and_weights():
    cfg = SPCATrackConfig()
    assert cfg.cue_weights == (0.50,0.20,0.10,0.20)
    assert abs(sum(cfg.cue_weights)-1) < 1e-8


def test_paper_objective_defaults():
    cfg = SPCATrackConfig()
    assert (cfg.lambda_con, cfg.lambda_nwd, cfg.lambda_tcr, cfg.lambda_mifd, cfg.lambda_hard) == (0.10, 0.50, 0.10, 0.05, 0.10)
    assert cfg.mifd_alpha == 0.30 and cfg.tau_soft == 0.10
    assert cfg.hard_negative_ratio == 0.02 and cfg.hard_negative_margin == 0.30
    assert cfg.tau_con == 0.10 and cfg.eta_nwd == 0.50
    assert cfg.window_size == 30 and cfg.context_history == 10 and cfg.max_lost == 5
    assert cfg.shared_dim == 128 and cfg.beta == 0.20 and cfg.imgsz == 640 and cfg.amp is True
    assert (cfg.token_embedding_dim, cfg.language_hidden_dim, cfg.context_dim) == (64, 128, 128)
    assert cfg.enabled_detector_components() == {"sps", "tcr", "mifd", "bns-hm", "msa-nwd"}


def test_detector_ablation_switch_matrix():
    expected = {
        "base": set(),
        "w/o-tcr": {"sps", "mifd", "bns-hm", "msa-nwd"},
        "w/o-mifd": {"sps", "tcr", "bns-hm", "msa-nwd"},
        "w/o-stcr": {"sps", "bns-hm", "msa-nwd"},
        "w/o-bns-hm": {"sps", "tcr", "mifd", "msa-nwd"},
        "w/o-msa-nwd": {"sps", "tcr", "mifd", "bns-hm"},
        "w/o-sps": {"msa-nwd"},
    }
    for name, components in expected.items():
        cfg = SPCATrackConfig(detector_ablation=name)
        cfg.validate()
        assert cfg.enabled_detector_components() == components
    SPCATrackConfig(imgsz=1280).validate()


def test_structured_mapper_matches_manuscript_dimensions():
    tokenizer = StructuredPromptTokenizer()
    prompt = WindowSummary(2, (1, 0, 1), 0.25, 0.50, "horizontal", 0.74, 0.1, 0.0).prompt()
    ids = tokenizer.encode(prompt)
    assert tokenizer.unk_id not in ids.tolist()
    encoder = StructuredPromptEncoder(device="cpu")
    assert encoder.token_embedding.embedding_dim == 64
    assert isinstance(encoder.mlp[1], torch.nn.GELU)
    assert encoder.mlp[0].out_features == 128 and encoder.mlp[2].out_features == 128
    assert encoder.encode(prompt).shape == (128,)


def test_prompt_controls_are_causal_and_deterministic():
    current = WindowSummary(9, (3, 2, 4), 0.8, 0.7, "vertical", 0.9, 0.0, 0.2)
    previous = [
        WindowSummary(2, (1, 0, 1), 0.1, 0.2, "horizontal", 0.6, 0.1, 0.0),
        WindowSummary(3, (0, 1, 2), 0.3, 0.4, "stationary", 0.7, 0.0, 0.0),
    ]
    neutral = WindowSummary.empty().prompt()
    assert controlled_prompt(current, previous, "constant", np.random.default_rng(0)) == neutral
    assert controlled_prompt(current, [], "shuffled-window", np.random.default_rng(0)) == neutral
    a = controlled_prompt(current, previous, "shuffled-window", np.random.default_rng(7))
    b = controlled_prompt(current, previous, "shuffled-window", np.random.default_rng(7))
    assert a == b and a in {summary.prompt() for summary in previous}
    mixed = controlled_prompt(current, previous, "attribute-permuted", np.random.default_rng(3))
    assert mixed != current.prompt()


def test_pair_cue_and_prompt_attribute_ablations():
    summary = WindowSummary(2, (1, 0, 1), 0.25, 0.50, "horizontal", 0.74, 0.1, 0.0)
    prompt = summary.prompt(("qs",))
    assert "identities were observed" not in prompt and "States:" in prompt

    cfg = SPCATrackConfig(disabled_cues=("size",))
    tracker = LGATracker((640, 480), DummyEncoder(), PairContextAdapter(128, 128), cfg, device="cpu")
    tracker.update([Detection(np.array([10, 10, 20, 20], np.float32), 0.9)])
    active = next(iter(tracker.tracks.values()))
    cues = tracker.pairwise_cues(Detection(np.array([12, 10, 22, 20], np.float32), 0.9, state=0), active)
    assert cues[1] == 0.0


def test_geometry_only_has_no_window_context():
    cfg = SPCATrackConfig(window_size=2, context_variant="geometry-only")
    tracker = LGATracker((640, 480), None, None, cfg, device="cpu")
    tracker.update([Detection(np.array([10, 10, 20, 20], np.float32), 0.9)])
    tracker.update([Detection(np.array([12, 10, 22, 20], np.float32), 0.9)])
    assert tracker.causal_context is None
    assert not tracker.completed_summaries
