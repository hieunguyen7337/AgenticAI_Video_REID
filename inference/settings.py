from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSettings:
    name: str = "ViT-B-16"
    device: str = "cuda"
    pretrain_path: str = ""
    stride_size: tuple[int, int] = (16, 16)
    sie_camera: bool = True
    sie_view: bool = False
    sie_coe: float = 1.0


@dataclass(frozen=True)
class InputSettings:
    size: tuple[int, int] = (256, 128)
    seq_len: int = 8


@dataclass(frozen=True)
class TestSettings:
    neck_feat: str = "before"


@dataclass(frozen=True)
class InferenceSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    input: InputSettings = field(default_factory=InputSettings)
    test: TestSettings = field(default_factory=TestSettings)


DEFAULT_SETTINGS = InferenceSettings()


def build_settings(device=None, clip_pretrain_path=None):
    model_settings = ModelSettings(
        device=device or DEFAULT_SETTINGS.model.device,
        pretrain_path=clip_pretrain_path or DEFAULT_SETTINGS.model.pretrain_path,
    )
    return InferenceSettings(
        model=model_settings,
        input=DEFAULT_SETTINGS.input,
        test=DEFAULT_SETTINGS.test,
    )
