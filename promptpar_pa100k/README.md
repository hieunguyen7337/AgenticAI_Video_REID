# `promptpar_pa100k`

Standalone PromptPAR inference package for the PA100k pedestrian attribute recognition task.

This package was extracted from the `PromptPAR` subfolder of the larger `OpenPAR` repository and trimmed down so it can be moved elsewhere and used for inference without depending on the rest of the repo.

## What This Package Contains

- A single public Python function: `infer_attributes(...)`
- The PA100k attribute vocabulary
- The tokenizer asset used by CLIP text encoding
- The packaged PA100k checkpoint
- Inference-only model code for:
  - CLIP visual/text encoding
  - PromptPAR multimodal fusion
  - attribute scoring and thresholding

## What This Package Does Not Require

- No dataset pickles
- No `train.py`, `test_example.py`, or `test_in_custom.py`
- No `argparse` flags at runtime
- No external download of CLIP weights
- No external `jx_vit_base_p16_224-80ecf9dd.pth`
- No dependency on the rest of the `OpenPAR` repo after this folder is copied out

## Package Layout

- [__init__.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\__init__.py)
  Exports the public entrypoint.
- [inference.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\inference.py)
  Loads assets, prepares input, runs inference, and formats output.
- [config.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\config.py)
  Fixed inference configuration and asset paths.
- [metadata.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\metadata.py)
  PA100k attribute names and output mapping helper.
- [tokenizer.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\tokenizer.py)
  Standalone tokenizer used for text prompts.
- [modeling_clip.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\modeling_clip.py)
  Inference-only CLIP model code.
- [modeling_fusion.py](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\modeling_fusion.py)
  PromptPAR fusion and attribute classifier code.
- [assets/bpe_simple_vocab_16e6.txt.gz](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\assets\bpe_simple_vocab_16e6.txt.gz)
  Tokenizer vocabulary file.
- [weights/PA100k_Checkpoint.pth](F:\document\Agentic AI_video_re_identification\OpenPAR\PromptPAR\promptpar_pa100k\weights\PA100k_Checkpoint.pth)
  Packaged PA100k inference checkpoint.

## Fixed Inference Configuration

This standalone package is PA100k-specific and uses fixed settings baked into the code:

- input size: `224 x 224`
- default threshold: `0.45`
- `use_div=True`
- `use_vismask=True`
- `use_gl=True`
- `use_textprompt=True`
- `use_mm_former=True`
- `mm_layers=1`
- `div_num=4`
- `overlap_row=2`
- `text_prompt=3`
- `vis_prompt=50`
- `vis_depth=24`
- attribute count: `26`

These values are not passed in through CLI flags. They are part of the package behavior.

## Dependencies

The target Python environment should have:

- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `ftfy`
- `regex`

## Public API

Import:

```python
from promptpar_pa100k import infer_attributes
```

Function:

```python
infer_attributes(image, threshold=0.45, device=None, return_probs=False)
```

### Parameters

- `image`
  The image to run inference on.
- `threshold`
  Sigmoid cutoff used to turn probabilities into predicted attribute names.
- `device`
  Optional device override, such as `"cpu"`, `"cuda"`, or `torch.device(...)`.
- `return_probs`
  If `True`, includes an extra `probabilities` field in the returned dict.

### Accepted Input Types

`image` can be any one of:

- `str`
  File path to an image.
- `pathlib.Path`
  File path object.
- `PIL.Image.Image`
  Loaded PIL image.
- `numpy.ndarray`
  Accepted shapes:
  - `H x W`
  - `H x W x 1`
  - `H x W x 3`
  - `H x W x 4`
- `torch.Tensor`
  Accepted shapes:
  - `C x H x W`
  - `H x W x C`
  - `1 x C x H x W`

Tensor inputs are treated as a single image. If values are above `1.0`, they are assumed to be in `0..255` range and scaled down.

## Preprocessing

Non-tensor inputs are preprocessed as:

```python
Resize((224, 224)) -> ToTensor() -> Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

Tensor inputs are resized to `224 x 224` and normalized the same way.

## Output Format

The function returns a Python `dict` with:

- `attributes`
  A list of predicted PA100k attribute names whose probability is above `threshold`.
- `scores`
  An ordered mapping from every PA100k attribute name to its sigmoid probability.
- `threshold`
  The threshold actually used.

If `return_probs=True`, the dict also contains:

- `probabilities`
  Same ordered mapping as `scores`

### Example Return Value

```python
{
    "attributes": [
        "A female pedestrian",
        "A pedestrian with a backpack"
    ],
    "scores": OrderedDict([
        ("A female pedestrian", 0.98),
        ("A pedestrian over the age of 60", 0.02),
        ...
    ]),
    "threshold": 0.45
}
```

## Example Usage

### 1. Basic path input

```python
from promptpar_pa100k import infer_attributes

result = infer_attributes("person.jpg")
print(result["attributes"])
```

### 2. Force CPU

```python
from promptpar_pa100k import infer_attributes

result = infer_attributes("person.jpg", device="cpu")
print(result["scores"])
```

### 3. Use a PIL image

```python
from PIL import Image
from promptpar_pa100k import infer_attributes

image = Image.open("person.jpg").convert("RGB")
result = infer_attributes(image)
```

### 4. Use a NumPy array

```python
import numpy as np
from PIL import Image
from promptpar_pa100k import infer_attributes

array = np.array(Image.open("person.jpg").convert("RGB"))
result = infer_attributes(array)
```

### 5. Use a tensor

```python
import torchvision.io as io
from promptpar_pa100k import infer_attributes

tensor = io.read_image("person.jpg")
result = infer_attributes(tensor)
```

## How Model Loading Works

- The model is loaded lazily on the first call to `infer_attributes(...)`
- The loaded model is cached in memory for later calls
- If `device=None`, the package selects:
  - CUDA if available
  - otherwise CPU

This means the first call is slower than later calls.

## Checkpoint Format Expectations

The packaged checkpoint is expected to contain:

- `ViT_model` or `clip_model`
- `model_state_dict`

The current PA100k checkpoint in this package contains:

- `ViT_model`
- `model_state_dict`
- `epoch`

The loader also normalizes `module.` prefixes and maps `visual_embed.*` to `vis_embed.*` if needed for checkpoint compatibility.

## Attributes

This package predicts the 26 PA100k attributes:

1. `A female pedestrian`
2. `A pedestrian over the age of 60`
3. `A pedestrian between the ages of 18 and 60`
4. `A pedestrian under the age of 18`
5. `A pedestrian seen from the front`
6. `A pedestrian seen from the side`
7. `A pedestrian seen from the back`
8. `A pedestrian wearing a hat`
9. `A pedestrian wearing glasses`
10. `A pedestrian with a handbag`
11. `A pedestrian with a shoulder bag`
12. `A pedestrian with a backpack`
13. `A pedestrian holding objects in front`
14. `A pedestrian in short-sleeved upper wear`
15. `A pedestrian in long-sleeved upper wear`
16. `A pedestrian in stride upper wear`
17. `A pedestrian in upper wear with a logo`
18. `A pedestrian in plaid upper wear`
19. `A pedestrian in splice upper wear`
20. `A pedestrian in striped lower wear`
21. `A pedestrian in patterned lower wear`
22. `A pedestrian in a long coat`
23. `A pedestrian in trousers`
24. `A pedestrian in shorts`
25. `A pedestrian in skirts and dresses`
26. `A pedestrian wearing boots`

## Portability

To move this package elsewhere, copy the whole `promptpar_pa100k` folder and keep the internal structure unchanged:

- `assets/`
- `weights/`
- Python source files

The package resolves assets relative to its own file location, so it does not depend on the current working directory.

## Limitations

- PA100k only
- Inference only
- No batch public API at the package root
- No training support
- No metrics/evaluation helpers
- No checkpoint auto-download
- Assumes the installed environment provides the required Python dependencies

## Minimal Call

```python
from promptpar_pa100k import infer_attributes

result = infer_attributes("image.jpg")
```

That is the only function call required to run inference.
