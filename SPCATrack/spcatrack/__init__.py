from .config import SPCATrackConfig
from .types import Detection, Track, TrackObservation
from .sps import (
    SPSBranch,
    ISPSAuxiliaryObjective,
    MSANWD,
    bns_hard_negative_loss,
    foreground_background_contrastive_loss,
    mifd_loss,
    temporal_consistency_loss,
)
from .context import NumericMLPEncoder, StructuredPromptEncoder, StructuredPromptTokenizer
from .lgatracker import (
    LGATracker,
    PairContextAdapter,
    WindowSummary,
    controlled_prompt,
    summarize_window,
)
from .pipeline import SPCATrackPipeline

__all__ = [
    "SPCATrackConfig", "Detection", "Track", "TrackObservation",
    "SPSBranch", "ISPSAuxiliaryObjective", "MSANWD",
    "bns_hard_negative_loss", "foreground_background_contrastive_loss",
    "mifd_loss", "temporal_consistency_loss", "LGATracker",
    "PairContextAdapter", "WindowSummary", "summarize_window",
    "controlled_prompt", "StructuredPromptTokenizer", "StructuredPromptEncoder",
    "NumericMLPEncoder", "SPCATrackPipeline",
]
from .objective import compose_detector_objective, FullDetectorLoss
__all__ += ["compose_detector_objective", "FullDetectorLoss"]
