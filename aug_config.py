"""Ablatable augmentation configuration for SuperSimpleNet.

All augmentation is off by default (enabled=False) so a clean, augmentation-free
baseline is reachable (identical to the current Resize+Normalize behaviour). A
config JSON drives exactly which families run, enabling single-family ablations.

Schema is kept identical to the SK-RD4AD counterpart so both repos share the same
config vocabulary. In SuperSimpleNet the geometric families (flips, affine,
dynamic_crop) are applied jointly to image + mask(s); see augment_ssn.py.
"""
import json
from dataclasses import dataclass, asdict

AUG_SCHEMA_VERSION = "1.0"


@dataclass
class AugConfig:
    enabled: bool = False              # master switch: False => NO augmentation at all
    dynamic_crop: bool = False         # content-based crop+resize (joint on image/mask)
    equalize_p: float = 0.0            # probability of histogram equalization (image-only)
    hflip_p: float = 0.0               # geometric (joint on image/mask)
    vflip_p: float = 0.0               # geometric (joint on image/mask)
    affine_deg: float = 0.0            # +/- degrees; 0 => no rotation (joint)
    affine_translate: float = 0.0      # fraction; 0 => no translation (joint)
    affine_scale: tuple = (1.0, 1.0)   # (1.0, 1.0) => no scaling (joint)
    brightness: float = 0.0            # photometric (image-only)
    contrast: float = 0.0              # photometric (image-only)
    saturation: float = 0.0            # photometric (image-only)
    hue: float = 0.0                   # photometric (image-only)
    grayscale_p: float = 0.0           # photometric (image-only)
    blur_p: float = 0.0                # probability of Gaussian blur (image-only)
    blur_kernel: int = 3
    blur_sigma: tuple = (0.1, 1.0)
    speckle_std: float = 0.0           # additive Gaussian noise std on [0,1]; 0 => off
    seed: int = 0                      # recorded for provenance/logging only (RNG comes
                                       # from the global seed_everything(config["seed"]))

    @classmethod
    def from_json(cls, path: str) -> "AugConfig":
        with open(path, "r") as f:
            data = json.load(f)
        data.pop("schema_version", None)
        # JSON arrays load as lists; restore tuples for torchvision.
        for k in ("affine_scale", "blur_sigma"):
            if k in data and isinstance(data[k], list):
                data[k] = tuple(data[k])
        return cls(**data)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema_version"] = AUG_SCHEMA_VERSION
        return d
