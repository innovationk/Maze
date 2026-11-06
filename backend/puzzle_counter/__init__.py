"""Zero-shot puzzle-piece counter: Grounding DINO (detection) + SAM 2 (segmentation)."""

from .pipeline import CounterConfig, PieceCounter

__all__ = ["CounterConfig", "PieceCounter"]
