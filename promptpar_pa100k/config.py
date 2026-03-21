from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptPARConfig:
    image_size: int = 224
    default_threshold: float = 0.45
    use_div: bool = True
    use_vismask: bool = True
    use_gl: bool = True
    use_textprompt: bool = True
    use_mm_former: bool = True
    mm_layers: int = 1
    div_num: int = 4
    overlap_row: int = 2
    text_prompt: int = 3
    vis_prompt: int = 50
    vis_depth: int = 24
    attr_count: int = 26
    context_length: int = 77
    checkpoint_name: str = "PA100k_Checkpoint.pth"
    tokenizer_name: str = "bpe_simple_vocab_16e6.txt.gz"

    @property
    def package_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def assets_dir(self) -> Path:
        return self.package_root / "assets"

    @property
    def weights_dir(self) -> Path:
        return self.package_root / "weights"

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / self.checkpoint_name

    @property
    def tokenizer_path(self) -> Path:
        return self.assets_dir / self.tokenizer_name


DEFAULT_CONFIG = PromptPARConfig()
