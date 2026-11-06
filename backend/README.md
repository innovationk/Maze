# Puzzle piece counter (zero-shot baseline)

Counts jigsaw puzzle pieces in a picture, ignoring other objects, with two pretrained models and no training:

- **Grounding DINO** (`IDEA-Research/grounding-dino-base`, Apache-2.0) finds where the puzzle material is from the text prompt `"a puzzle piece."`.
- **SAM 2.1** (`facebook/sam2.1-hiera-small`, Apache-2.0) segments the individual pieces.

## Setup

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads about 1 GB of weights from Hugging Face into `~/.cache/huggingface`.

## Usage

```bash
# Count the pieces in every image of a folder and write JSON
.venv/bin/python count_pieces.py ../TESTS -o results/tests.json

# Also save annotated images (one colour + number per counted piece) and per-piece details
.venv/bin/python count_pieces.py ../TESTS -o results/tests.json --viz-dir results/viz --details

# Evaluate against a CSV with columns label,count
.venv/bin/python count_pieces.py ../dataset --labels ../dataset/dataset.csv -o results/dataset.json
```

Output:

```json
{
  "config": { "...": "every parameter used" },
  "results": [
    { "image": "photo.webp", "count": 142, "typical_piece_area": 2730, "seconds": 88.0 }
  ],
  "metrics": { "images": 30, "mae": 0.0, "mape": 0.0, "exact": 0 }
}
```

`expected`, `error` and `metrics` appear only with `--labels`. Per-piece boxes appear only with `--details`.

## How it works

See the docstring in [puzzle_counter/pipeline.py](puzzle_counter/pipeline.py). In short:

1. Grounding DINO returns boxes around loose pieces and around assembled areas. Their union is the
   *puzzle region*; anything outside it (other objects, background) is ignored.
2. SAM 2 segments every piece-sized box, plus a 32×32 grid of point prompts inside the puzzle region.
   The point grid is what splits an assembled puzzle into pieces: Grounding DINO alone only returns
   one box for a whole assembled puzzle.
3. Masks are filtered on SAM's quality and stability scores, de-duplicated, and kept only if their
   area is close to the *typical piece area* (the most common mask size, since pieces of one puzzle
   have similar sizes).
4. The count is the number of masks left after overlap suppression.

## Known limitations

- **Assembled puzzles with a printed picture.** When neighbouring pieces have the same colours, SAM 2
  follows the colour regions of the picture instead of the thin cut lines, so the assembled part is
  under-counted. Puzzles whose pieces have contrasting colours, and loose pieces, work much better.
- **Speed.** On a CPU (Intel i7) an image takes 1 to 2 minutes, mostly SAM 2's mask decoder on the
  point grid. `--points-per-side 24` is faster; a GPU (`--device cuda`) is much faster.
- **Very dense puzzles** (500+ pieces) need a denser point grid and higher-resolution masks than the defaults.

The next step to go beyond this baseline is to annotate images and fine-tune a detector or
instance-segmentation model (e.g. RF-DETR), using this pipeline to pre-annotate.
