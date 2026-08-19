"""GPU augmentation for SuperSimpleNet, ablatable via AugConfig.

Injected in the training loop on GPU tensors in [0,1], AFTER resize and BEFORE
ImageNet normalization. Unlike SK-RD4AD, SuperSimpleNet trains with segmentation
targets, so augmentation is split by geometry:

  * Photometric families (ColorJitter, grayscale, blur, equalize, speckle) act on
    the IMAGE only -- they do not move pixels, so the mask stays valid.
  * Geometric families (flips, affine, dynamic_crop) act JOINTLY on the image and
    every mask-like tensor (segmentation mask + optional loss_mask), so the
    supervised target stays pixel-aligned with the image.

Geometric params are sampled once per batch (this matches how torchvision v2
transforms behave on a batched [B,C,H,W] tensor) and applied with per-tensor
interpolation/fill: bilinear + white fill for the image, nearest + 0 fill for the
binary mask, nearest + 1 fill for the loss_mask (1 == "normal" weight).

The augmentation RNG is the global torch RNG, made reproducible by the existing
seed_everything(config["seed"]) call; AugConfig.seed is provenance-only.
"""
from typing import Optional

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F_v2

from aug_config import AugConfig

_BILINEAR = v2.InterpolationMode.BILINEAR
_NEAREST = v2.InterpolationMode.NEAREST


def build_photometric_augmentation(cfg: AugConfig) -> Optional[v2.Compose]:
    """Return an image-only v2.Compose of photometric transforms, or None.

    Only families with a non-default value are added. Histogram equalization and
    speckle noise are handled separately in the training loop (they need uint8 /
    additive-noise handling), so they are NOT part of this Compose.
    """
    if not cfg.enabled:
        return None

    ops = []
    if cfg.brightness > 0 or cfg.contrast > 0 or cfg.saturation > 0 or cfg.hue > 0:
        ops.append(v2.ColorJitter(
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            saturation=cfg.saturation,
            hue=cfg.hue,
        ))
    if cfg.grayscale_p > 0:
        ops.append(v2.RandomGrayscale(p=cfg.grayscale_p))
    if cfg.blur_p > 0:
        ops.append(v2.RandomApply(
            [v2.GaussianBlur(kernel_size=cfg.blur_kernel, sigma=tuple(cfg.blur_sigma))],
            p=cfg.blur_p,
        ))
    return v2.Compose(ops) if ops else None


def equalize_image(image: torch.Tensor) -> torch.Tensor:
    """Histogram-equalize a [0,1] float image batch (image-only, via uint8)."""
    img_uint8 = (image * 255.0).to(torch.uint8)
    return F_v2.equalize(img_uint8).to(torch.float32) / 255.0


class GeometricAugmentor:
    """Applies flips + affine jointly to image and mask-like tensors.

    Sampled once per batch. Call with the image and any number of mask-like
    tensors; pass each mask as (tensor, fill_value) so borders introduced by the
    affine get the correct value (0 for the binary mask, 1 for the loss_mask).
    """

    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.active = cfg.enabled and (
            cfg.hflip_p > 0
            or cfg.vflip_p > 0
            or cfg.affine_deg > 0
            or cfg.affine_translate > 0
            or tuple(cfg.affine_scale) != (1.0, 1.0)
        )

    def __call__(self, image, masks):
        """Transform ``image`` and ``masks`` in place-consistent fashion.

        Args:
            image: [B, C, H, W] float image in [0, 1].
            masks: list of (tensor, fill_value) tuples, each [B, 1, H, W].
        Returns:
            (image, [tensor, ...]) with the same geometric transform applied.
        """
        if not self.active:
            return image, [m for m, _ in masks]

        cfg = self.cfg
        out_masks = [m for m, _ in masks]

        # --- flips (per batch) ---
        if cfg.hflip_p > 0 and torch.rand(1).item() < cfg.hflip_p:
            image = F_v2.horizontal_flip(image)
            out_masks = [F_v2.horizontal_flip(m) for m in out_masks]
        if cfg.vflip_p > 0 and torch.rand(1).item() < cfg.vflip_p:
            image = F_v2.vertical_flip(image)
            out_masks = [F_v2.vertical_flip(m) for m in out_masks]

        # --- affine (per batch) ---
        if cfg.affine_deg > 0 or cfg.affine_translate > 0 or tuple(cfg.affine_scale) != (1.0, 1.0):
            h, w = image.shape[-2], image.shape[-1]
            angle = (torch.rand(1).item() * 2.0 - 1.0) * cfg.affine_deg
            max_dx = cfg.affine_translate * w
            max_dy = cfg.affine_translate * h
            tx = int(round((torch.rand(1).item() * 2.0 - 1.0) * max_dx))
            ty = int(round((torch.rand(1).item() * 2.0 - 1.0) * max_dy))
            s_min, s_max = cfg.affine_scale
            scale = s_min + torch.rand(1).item() * (s_max - s_min)

            image = F_v2.affine(
                image, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                interpolation=_BILINEAR, fill=1.0,
            )
            new_masks = []
            for (mask_out, (_, fill)) in zip(out_masks, masks):
                new_masks.append(F_v2.affine(
                    mask_out, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                    interpolation=_NEAREST, fill=float(fill),
                ))
            out_masks = new_masks

        return image, out_masks


def dynamic_crop_joint(image, masks, padding: int = 30):
    """Content-based per-sample crop+resize, applied jointly to image and masks.

    Mirrors SK-RD4AD's apply_dynamic_crop_gpu (dark-region bounding box) but also
    crops every mask-like tensor with the same box so alignment is preserved.

    Args:
        image: [B, C, H, W] float image in [0, 1].
        masks: list of (tensor, fill_value) tuples ([B, 1, H, W]); fill unused here
            (crop preserves values), kept for a uniform call signature.
    Returns:
        (image, [tensor, ...]) cropped and resized back to [H, W].
    """
    B, C, H, W = image.shape
    out_img = []
    out_masks = [[] for _ in masks]

    gray = image.mean(dim=1)
    is_dark = gray < 0.94

    for i in range(B):
        coords = torch.nonzero(is_dark[i])
        if coords.numel() == 0:
            out_img.append(image[i])
            for j, (m, _) in enumerate(masks):
                out_masks[j].append(m[i])
            continue

        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values
        size = torch.maximum(y_max - y_min, x_max - x_min)
        cy = y_min + (y_max - y_min) // 2
        cx = x_min + (x_max - x_min) // 2

        y1 = torch.clamp(cy - size // 2 - padding, min=0)
        y2 = torch.clamp(cy + size // 2 + padding, max=H)
        x1 = torch.clamp(cx - size // 2 - padding, min=0)
        x2 = torch.clamp(cx + size // 2 + padding, max=W)

        crop_img = image[i:i + 1, :, y1:y2, x1:x2]
        out_img.append(
            F_v2.resize(crop_img, size=[H, W], interpolation=_BILINEAR, antialias=True).squeeze(0)
        )
        for j, (m, _) in enumerate(masks):
            crop_m = m[i:i + 1, :, y1:y2, x1:x2]
            out_masks[j].append(
                F_v2.resize(crop_m, size=[H, W], interpolation=_NEAREST).squeeze(0)
            )

    image = torch.stack(out_img, dim=0)
    stacked = [torch.stack(m, dim=0) for m in out_masks]
    return image, stacked
