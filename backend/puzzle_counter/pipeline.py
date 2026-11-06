"""Count puzzle pieces in an image with Grounding DINO + SAM 2.

Why not just count Grounding DINO boxes?
    Grounding DINO is good at finding *loose* pieces, but on an assembled (or partly assembled)
    puzzle it returns a single "puzzle" box, whatever the prompt, threshold or tiling.

Pipeline
--------
1. Detect (Grounding DINO, prompt "a puzzle piece."): boxes around loose pieces and around
   assembled areas. Their union is the "puzzle region"; everything outside it (other objects,
   background) is ignored.
2. Segment (SAM 2), from two kinds of prompts:
   - each piece-sized detection box  -> one mask,
   - a regular grid of points inside the puzzle region -> 3 candidate masks per point
     (SAM's sub-part / object / group hypotheses). This is what splits assembled puzzles
     into individual pieces.
3. Filter candidates: SAM quality score, mask stability, mask lying inside the puzzle region.
4. Estimate the typical piece area (mode of the candidate areas: pieces of one puzzle have
   similar sizes) and keep masks whose area is close to it. This rejects sub-parts (a printed
   motif on a piece) and blobs of several pieces.
5. Greedy mask-NMS on pixel overlap. The count is the number of masks left.

All mask work happens on SAM 2's 256x256 low-resolution masks (SAM 2 resizes every image to a
1024x1024 square, so low-res masks map linearly onto the image).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import nms

# torch 2.2.2 is the last release for Intel Macs; it lacks torch.compiler.is_compiling (added in 2.3),
# which transformers' fast image processors call.
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

from transformers import (  # noqa: E402
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

log = logging.getLogger(__name__)


@dataclass
class CounterConfig:
    detector_id: str = "IDEA-Research/grounding-dino-base"
    segmenter_id: str = "facebook/sam2.1-hiera-small"
    prompt: str = "a puzzle piece."

    # --- detection (Grounding DINO)
    box_threshold: float = 0.20
    text_threshold: float = 0.20
    box_nms_iou: float = 0.5
    group_min_children: int = 2  # a box enclosing >= this many other boxes is a group, not a piece
    containment: float = 0.85  # fraction of a child box that must lie inside its parent

    # --- segmentation (SAM 2)
    points_per_side: int = 32  # point grid over the whole image; only points inside the puzzle region are used
    pred_iou_threshold: float = 0.75  # SAM's own mask-quality estimate
    stability_threshold: float = 0.85  # IoU between the mask thresholded at -1 and +1 logits
    inside_region: float = 0.90  # fraction of a point mask that must lie inside the puzzle region

    # --- piece-size filtering and de-duplication
    size_band: tuple[float, float] = (0.4, 1.8)  # accepted area range, relative to the typical piece area
    box_mask_max_ratio: float = 2.5  # box-prompted masks larger than this x typical area are groups
    mask_overlap: float = 0.5  # drop a mask if this fraction of it is already covered by kept masks
    duplicate_iou: float = 0.85  # candidate masks this similar are the same hypothesis (merged before voting)

    batch_size: int = 64
    device: str | None = None


@dataclass
class Piece:
    box: list[float]  # [x1, y1, x2, y2] in image pixels, from the mask
    score: float  # combined quality score used for ranking
    area: int  # approximate mask area in image pixels
    source: str  # "box" (Grounding DINO box prompt) or "point" (SAM point-grid prompt)


@dataclass
class CountResult:
    count: int
    pieces: list[Piece] = field(default_factory=list)
    typical_piece_area: int | None = None
    masks: np.ndarray | None = None  # (count, 256, 256) bool low-res masks, for visualisation


@dataclass
class _Candidates:
    masks: np.ndarray  # (N, 256, 256) bool
    scores: np.ndarray  # (N,)
    source: list[str]

    @staticmethod
    def empty(size: int) -> "_Candidates":
        return _Candidates(np.zeros((0, size, size), dtype=bool), np.zeros(0, dtype=np.float32), [])

    def __len__(self) -> int:
        return len(self.scores)


def _pick_device(requested: str | None) -> str:
    if requested:
        return requested
    # MPS is opt-in (--device mps): Grounding DINO uses ops that older macOS versions don't support.
    return "cuda" if torch.cuda.is_available() else "cpu"


class PieceCounter:
    LOW_RES = 256  # SAM 2 low-resolution mask size

    def __init__(self, config: CounterConfig | None = None):
        self.cfg = config or CounterConfig()
        self.device = _pick_device(self.cfg.device)
        log.info("Loading %s and %s on %s", self.cfg.detector_id, self.cfg.segmenter_id, self.device)

        self.det_processor = AutoProcessor.from_pretrained(self.cfg.detector_id)
        self.det_model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.cfg.detector_id).to(self.device).eval()
        )
        self.seg_processor = Sam2Processor.from_pretrained(self.cfg.segmenter_id)
        self.seg_model = Sam2Model.from_pretrained(self.cfg.segmenter_id).to(self.device).eval()

    # ------------------------------------------------------------------ public API

    @torch.inference_mode()
    def count(self, image: Image.Image) -> CountResult:
        image = image.convert("RGB")
        width, height = image.size

        boxes, scores = self._detect(image)
        if len(boxes) == 0:
            return CountResult(count=0)
        is_group = self._group_boxes(boxes)
        log.debug("detections: %d (%d groups)", len(boxes), int(is_group.sum()))

        region = self._region_mask(boxes, width, height)
        pixel_values = self.seg_processor(images=image, return_tensors="pt")["pixel_values"]
        embeddings = self.seg_model.get_image_embeddings(pixel_values.to(self.device))

        box_cands = self._segment_boxes(embeddings, boxes[~is_group], scores[~is_group], width, height)
        point_cands = self._dedupe(self._segment_points(embeddings, region, width, height))
        log.debug("candidates: %d from boxes, %d from points", len(box_cands), len(point_cands))

        typical_area = self._typical_area(point_cands, box_cands)
        if typical_area is None:
            return CountResult(count=0)

        kept_masks, kept_scores, kept_source = self._select(box_cands, point_cands, typical_area)

        scale = (width * height) / (self.LOW_RES * self.LOW_RES)
        pieces = [
            Piece(
                box=self._mask_box(m, width, height),
                score=round(float(s), 4),
                area=int(m.sum() * scale),
                source=src,
            )
            for m, s, src in zip(kept_masks, kept_scores, kept_source)
        ]
        return CountResult(
            count=len(pieces),
            pieces=pieces,
            typical_piece_area=int(typical_area * scale),
            masks=np.stack(kept_masks) if kept_masks else None,
        )

    # ------------------------------------------------------------------ detection

    def _detect(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.det_processor(images=image, text=self.cfg.prompt, return_tensors="pt").to(self.device)
        outputs = self.det_model(**inputs)
        result = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.cfg.box_threshold,
            text_threshold=self.cfg.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes, scores = result["boxes"].float().cpu(), result["scores"].float().cpu()
        if len(boxes) == 0:
            return boxes, scores
        keep = nms(boxes, scores, self.cfg.box_nms_iou)
        return boxes[keep], scores[keep]

    def _group_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """True for boxes that enclose several other boxes (an assembled area, not a single piece)."""
        n = len(boxes)
        if n < self.cfg.group_min_children + 1:
            return torch.zeros(n, dtype=torch.bool)
        areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        lt = torch.max(boxes[:, None, :2], boxes[None, :, :2])
        rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]  # inter[i, j] = |box_i ∩ box_j|
        contains = (inter / areas[None, :].clamp(min=1e-6)) >= self.cfg.containment
        contains &= areas[None, :] < 0.7 * areas[:, None]
        contains.fill_diagonal_(False)
        return contains.sum(dim=1) >= self.cfg.group_min_children

    def _region_mask(self, boxes: torch.Tensor, width: int, height: int) -> np.ndarray:
        """Union of detection boxes, rasterised on the low-res mask grid."""
        r = self.LOW_RES
        region = np.zeros((r, r), dtype=bool)
        for x1, y1, x2, y2 in boxes.tolist():
            c1, c2 = int(math.floor(x1 / width * r)), int(math.ceil(x2 / width * r))
            r1, r2 = int(math.floor(y1 / height * r)), int(math.ceil(y2 / height * r))
            region[max(r1, 0) : min(r2, r), max(c1, 0) : min(c2, r)] = True
        return region

    # ------------------------------------------------------------------ segmentation

    def _decode(self, embeddings, width: int, height: int, multimask: bool, **prompt) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SAM 2 prompt encoder + mask decoder on precomputed image embeddings."""
        processed = self.seg_processor(original_sizes=[[height, width]], return_tensors="pt", **prompt)
        tensors = {k: v.to(self.device) for k, v in processed.items() if k.startswith("input_")}
        out = self.seg_model(image_embeddings=embeddings, multimask_output=multimask, **tensors)
        # pred_masks: (1, n_prompts, n_masks, 256, 256) logits; iou_scores: (1, n_prompts, n_masks)
        return out.pred_masks[0].float().cpu(), out.iou_scores[0].float().cpu()

    @staticmethod
    def _stability(logits: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
        high = (logits > offset).flatten(-2).sum(-1).float()
        low = (logits > -offset).flatten(-2).sum(-1).float()
        return high / low.clamp(min=1)

    def _segment_boxes(self, embeddings, boxes, scores, width: int, height: int) -> _Candidates:
        if len(boxes) == 0:
            return _Candidates.empty(self.LOW_RES)
        masks, mask_scores = [], []
        for start in range(0, len(boxes), self.cfg.batch_size):
            chunk = boxes[start : start + self.cfg.batch_size]
            logits, iou = self._decode(embeddings, width, height, multimask=False, input_boxes=[chunk.tolist()])
            masks.append((logits[:, 0] > 0).numpy())
            mask_scores.append(iou[:, 0].numpy() * scores[start : start + self.cfg.batch_size].numpy())
        all_masks = np.concatenate(masks)
        nonempty = all_masks.reshape(len(all_masks), -1).any(1)
        return _Candidates(all_masks[nonempty], np.concatenate(mask_scores)[nonempty], ["box"] * int(nonempty.sum()))

    def _segment_points(self, embeddings, region: np.ndarray, width: int, height: int) -> _Candidates:
        n, r = self.cfg.points_per_side, self.LOW_RES
        grid = (np.arange(n) + 0.5) / n  # normalised cell centres
        pts = [(gx, gy) for gy in grid for gx in grid if region[int(gy * r), int(gx * r)]]
        if not pts:
            return _Candidates.empty(r)

        region_t = torch.from_numpy(region)
        masks, mask_scores = [], []
        for start in range(0, len(pts), self.cfg.batch_size):
            chunk = pts[start : start + self.cfg.batch_size]
            input_points = [[[[gx * width, gy * height]] for gx, gy in chunk]]
            input_labels = [[[1] for _ in chunk]]
            logits, iou = self._decode(
                embeddings, width, height, multimask=True, input_points=input_points, input_labels=input_labels
            )
            logits, iou = logits.flatten(0, 1), iou.flatten(0, 1)  # (n_points * 3, 256, 256)
            binary = logits > 0
            area = binary.flatten(1).sum(1).float()
            inside = (binary & region_t).flatten(1).sum(1).float() / area.clamp(min=1)
            stability = self._stability(logits)
            ok = (
                (area > 0)
                & (iou >= self.cfg.pred_iou_threshold)
                & (stability >= self.cfg.stability_threshold)
                & (inside >= self.cfg.inside_region)
            )
            masks.append(binary[ok].numpy())
            mask_scores.append((iou * stability)[ok].numpy())

        all_masks = np.concatenate(masks)
        return _Candidates(all_masks, np.concatenate(mask_scores), ["point"] * len(all_masks))

    # ------------------------------------------------------------------ selection

    def _dedupe(self, cands: _Candidates) -> _Candidates:
        """Merge near-identical masks so each distinct hypothesis votes once in _typical_area.

        Without this, every grid point on an assembled puzzle proposes the same whole-puzzle mask,
        and that size wins the vote.
        """
        n = len(cands)
        if n < 2:
            return cands
        # IoU on 64x64 downsampled masks keeps the N x N computation cheap.
        small = cands.masks.reshape(n, 64, 4, 64, 4).any(axis=(2, 4)).reshape(n, -1)
        flat = torch.from_numpy(small).float()
        inter = flat @ flat.T
        area = flat.sum(1)
        iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1)
        duplicate = (iou > self.cfg.duplicate_iou).numpy()

        suppressed = np.zeros(n, dtype=bool)
        keep = []
        for i in np.argsort(-cands.scores):
            if suppressed[i]:
                continue
            keep.append(i)
            suppressed |= duplicate[i]
        keep = np.array(keep)
        return _Candidates(cands.masks[keep], cands.scores[keep], [cands.source[i] for i in keep])

    @staticmethod
    def _typical_area(primary: _Candidates, fallback: _Candidates) -> float | None:
        """Most frequent mask area (mode of a log-area histogram)."""
        cands = primary if len(primary) else fallback
        if not len(cands):
            return None
        areas = cands.masks.reshape(len(cands), -1).sum(1).astype(np.float64)
        log_a = np.log2(areas)
        bins = np.arange(log_a.min(), log_a.max() + 0.5, 0.25)
        if len(bins) < 2:
            return float(np.median(areas))
        hist, edges = np.histogram(log_a, bins=bins, weights=cands.scores)
        smoothed = np.convolve(hist, [1, 2, 1], mode="same")  # robust to a peak split across two bins
        peak = int(np.argmax(smoothed))
        lo, hi = edges[max(peak - 1, 0)], edges[min(peak + 2, len(edges) - 1)]
        in_peak = (log_a >= lo) & (log_a <= hi)
        return float(2 ** np.median(log_a[in_peak]))

    def _select(self, box_cands: _Candidates, point_cands: _Candidates, typical: float):
        lo, hi = self.cfg.size_band
        ordered: list[tuple[float, np.ndarray, str]] = []

        # Point masks: only piece-sized ones, ranked by quality and closeness to the typical size.
        for mask, score in zip(point_cands.masks, point_cands.scores):
            ratio = mask.sum() / typical
            if lo <= ratio <= hi:
                ordered.append((float(score) - 0.5 * abs(math.log2(ratio)), mask, "point"))
        ordered.sort(key=lambda t: t[0], reverse=True)

        # Box masks come after: they add loose pieces the point grid missed (e.g. much smaller or
        # larger because of perspective), but must not be whole-puzzle masks.
        box_list = [
            (float(s), m, "box")
            for m, s in zip(box_cands.masks, box_cands.scores)
            if lo * typical <= m.sum() <= self.cfg.box_mask_max_ratio * typical
        ]
        ordered += sorted(box_list, key=lambda t: t[0], reverse=True)

        occupied = np.zeros((self.LOW_RES, self.LOW_RES), dtype=bool)
        kept_masks, kept_scores, kept_source = [], [], []
        for score, mask, source in ordered:
            area = mask.sum()
            if (mask & occupied).sum() / area > self.cfg.mask_overlap:
                continue
            kept_masks.append(mask)
            kept_scores.append(score)
            kept_source.append(source)
            occupied |= mask
        return kept_masks, kept_scores, kept_source

    def _mask_box(self, mask: np.ndarray, width: int, height: int) -> list[float]:
        ys, xs = np.nonzero(mask)
        sx, sy = width / self.LOW_RES, height / self.LOW_RES
        return [
            round(float(xs.min() * sx), 1),
            round(float(ys.min() * sy), 1),
            round(float((xs.max() + 1) * sx), 1),
            round(float((ys.max() + 1) * sy), 1),
        ]


def render(image: Image.Image, result: CountResult) -> Image.Image:
    """Draw each counted piece (coloured mask + index) on a copy of the image."""
    image = image.convert("RGB")
    base = np.asarray(image).astype(np.float32)
    rng = np.random.default_rng(0)
    if result.masks is not None:
        full = F.interpolate(
            torch.from_numpy(result.masks)[:, None].float(), size=(image.height, image.width), mode="nearest"
        )[:, 0].bool().numpy()
        for mask in full:
            color = rng.integers(40, 255, size=3).astype(np.float32)
            base[mask] = 0.4 * base[mask] + 0.6 * color
    out = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for idx, piece in enumerate(result.pieces, start=1):
        x1, y1, x2, y2 = piece.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        draw.text((cx - 6, cy - 6), str(idx), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    draw.text((8, 8), f"count = {result.count}", fill=(255, 255, 0), stroke_width=3, stroke_fill=(0, 0, 0))
    return out
