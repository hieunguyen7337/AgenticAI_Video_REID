# TF-CLIP Inference Package

This package extracts the TF-CLIP test-time path into a small Python API.

It is intended to keep only the runtime pieces needed for video tracklet inference:

- visual feature extraction with the CLIP visual backbone
- temporal modeling with TMD
- dense clip sampling for long tracklets
- final sequence embedding generation
- cosine and Euclidean retrieval helpers

It intentionally removes training-only components such as CLIP-Memory, SSP, contrastive losses, optimizer code, and dataset-specific training loaders.

## Inference Pipeline Implemented Here

At a high level, the package follows this runtime path:

1. Input a tracklet as a list of frames.
2. Split the tracklet into dense clips of length `seq_len`.
3. Preprocess each frame with resize and normalization.
4. Run the clip through the CLIP visual backbone.
5. Run the visual tokens through TMD.
6. Aggregate each clip into one embedding.
7. Average embeddings across dense clips for long tracklets.
8. Optionally L2-normalize the final embedding.
9. Compare embeddings with cosine similarity or Euclidean distance.

This mirrors the repo's inference behavior more closely than a simplified paper-only description, especially for dense clip averaging.

## Code Structure

The package is split into a few small modules:

- `__init__.py`
  Exports the public API.
- `inferencer.py`
  High-level wrapper. Loads checkpoints, preprocesses tracklets, applies dense sampling, runs the model, and returns final embeddings.
- `model.py`
  Inference-only TF-CLIP model. Keeps the CLIP visual encoder, optional SIE camera/view embedding logic, TMD, and the bottlenecks used at test time.
- `backbone.py`
  Self-contained CLIP visual backbone code used at inference. This is what removes the old runtime dependency on CLIP weight download.
- `preprocess.py`
  Frame loading, resize, tensor conversion, and normalization.
- `sampling.py`
  Dense test-time clip splitting and repeat-last padding for short tracklets.
- `matching.py`
  Cosine similarity, Euclidean distance, ranking, and simple ReID metric evaluation.
- `types.py`
  Shared input type aliases.

## Public API

The main entrypoint is `TFClipInferencer`.

```python
from tfclip_inference import TFClipInferencer, cosine_similarity

inferencer = TFClipInferencer.from_checkpoint(
    "best_model.pth.tar",
    device="cuda",
    backbone="ViT-B-16",
    seq_len=8,
    image_size=(256, 128),
    stride_size=(16, 16),
    neck_feat="before",
    normalize=True,
)

embedding = inferencer.embed_tracklet(frame_paths, cam_id=0, view_id=0)
gallery_embeddings = inferencer.embed_tracklets(gallery_tracklets)
scores = cosine_similarity(embedding, gallery_embeddings)
```

### Main methods

- `TFClipInferencer.from_checkpoint(...)`
  Loads a TF-CLIP checkpoint and builds the inference-only model.
- `embed_tracklet(tracklet, cam_id=0, view_id=0)`
  Returns one final embedding for one person tracklet.
- `embed_tracklets(tracklets, cam_ids=None, view_ids=None)`
  Convenience wrapper for multiple tracklets.
- `embed_clip(clip_tensor, ...)`
  Runs a single already-preprocessed clip tensor of shape `(T, C, H, W)`.
- `cosine_similarity(...)`, `euclidean_distance(...)`, `rank_gallery(...)`, `evaluate_reid(...)`
  Lightweight retrieval helpers.

## Dense Sampling Behavior

Dense sampling is implemented to match the repo's test path:

- If `num_frames < seq_len`, the last frame is repeated until the clip reaches length `seq_len`.
- If `num_frames == seq_len`, one clip is used.
- If `num_frames > seq_len`, the tracklet is split into non-overlapping clips of length `seq_len`, and the tail clip is padded by repeating its last frame if needed.
- The final tracklet embedding is the mean of all clip embeddings.

## Feature Output Behavior

The model keeps the original TF-CLIP inference choices:

- `neck_feat="before"`
  Returns the concatenated feature used by the repo's common eval config.
- `neck_feat="after"`
  Returns the bottlenecked feature variant.
- `normalize=True`
  Applies final L2 normalization after dense clip averaging.

That last point matters: normalization is applied after averaging clip embeddings so long-tracklet behavior stays aligned with the original inference flow.

## Checkpoint Expectations

The loader is designed for TF-CLIP checkpoints saved from the original model.

It supports checkpoints where weights are stored as:

- raw state dicts
- nested `state_dict`
- nested `model`
- nested `model_state_dict`
- keys with or without the `module.` prefix

It also handles the checkpoint format used by the repo where the visual backbone is saved under `image_encoder.*` keys.

Training-only weights that do not belong to the inference package are ignored when loading.

## Preprocessing Details

The default preprocessing is based on the actual repo test loader behavior:

- resize to `(256, 128)` as `(height, width)`
- RGB conversion
- float tensor conversion in `[0, 1]`
- normalization with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`

The package accepts frames as:

- file paths
- PIL images
- tensors representing single frames

## Camera / View IDs

If the checkpoint was trained with SIE camera or view embeddings, you should pass the correct `cam_id` and `view_id` values at inference time.

If you omit them, the API defaults to `0`, which is convenient for simple embedding extraction but may not exactly match the original repo's evaluation behavior for SIE-enabled checkpoints.

## Can This Folder Be Moved Elsewhere?

Yes, the runtime package code is self-contained and can be moved to another location.

It should still run correctly if all of the following are true:

- you move the whole `tfclip_inference/` folder together
- `torch`, `numpy`, and `Pillow` are installed
- Python can import the package from its new parent directory, or you install it as a package
- you provide a compatible TF-CLIP checkpoint path

### Important caveats

- The package code inside `tfclip_inference/` does not depend on sibling repo files at runtime.
- The `tests/test_parity.py` file does still depend on the original repo, because it intentionally compares the extracted model against the original implementation.
- If you move only individual files instead of the full folder, the relative imports will break.
- If you package this elsewhere, keep the folder name `tfclip_inference` unless you also update the imports.

## Minimal Dependencies

The standalone package is designed to need only:

- `torch`
- `numpy`
- `Pillow`

That is much smaller than the full TF-CLIP training repo dependency surface.

## Example Project Layout After Moving

A minimal external layout can look like this:

```text
my_project/
  app.py
  tfclip_inference/
    __init__.py
    inferencer.py
    model.py
    backbone.py
    preprocess.py
    sampling.py
    matching.py
    types.py
    README.md
  checkpoints/
    best_model.pth.tar
```

Then `app.py` can import it with:

```python
from tfclip_inference import TFClipInferencer
```

## Tests

The package includes focused tests for:

- dense sampling
- preprocessing
- checkpoint key extraction
- clip and tracklet embedding wrapper behavior
- matching helpers
- parity with the original repo model path

The parity test is intentionally the least portable test because it validates that the extracted implementation still matches the original TF-CLIP code path.
