"""Image conversion pipeline tuned for thermal sticker printing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .errors import ImageProcessingError


def _load_grayscale(image_path: Path) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA"):
        composite = Image.new("RGBA", img.size, (255, 255, 255, 255))
        composite.alpha_composite(img.convert("RGBA"))
        return np.array(composite.convert("L"))
    return np.array(img.convert("L"))


def _normalize_contrast(img_np: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img_np, (2, 98))
    if p98 <= p2:
        return img_np
    return np.clip((img_np - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)


def _black_ratio(arr: np.ndarray) -> float:
    return float(np.mean(arr == 0))


def _pick_threshold_candidate(img_np: np.ndarray) -> np.ndarray:
    _, bw_otsu = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_adaptive = cv2.adaptiveThreshold(
        img_np,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    candidates = [bw_otsu, bw_adaptive]
    return min(
        candidates,
        key=lambda arr: abs(_black_ratio(arr) - 0.15)
        + (10 if _black_ratio(arr) < 0.005 or _black_ratio(arr) > 0.8 else 0),
    )


def _thermal_polarity_score(arr: np.ndarray, white_target: float) -> float:
    h, w = arr.shape
    mh = max(1, h // 20)
    mw = max(1, w // 20)
    corners = np.concatenate(
        [
            arr[:mh, :mw].ravel(),
            arr[:mh, -mw:].ravel(),
            arr[-mh:, :mw].ravel(),
            arr[-mh:, -mw:].ravel(),
        ]
    )

    white_ratio = float(np.mean(arr == 255))
    black_ratio = 1.0 - white_ratio
    corner_black_ratio = float(np.mean(corners == 0))

    score = abs(white_ratio - white_target)
    if corner_black_ratio > 0.25:
        score += 2.0
    if black_ratio < 0.003:
        score += 2.0
    if black_ratio > 0.60:
        score += 2.0
    return score


def convert_to_coloring_sticker(
    image_path: str | Path,
    output_path: str | Path | None = None,
    *,
    target_width: int,
    thermal_white_target: float,
) -> Path:
    source = Path(image_path)
    if not source.exists():
        raise ImageProcessingError(f"Input image does not exist: {source}")

    destination = Path(output_path) if output_path else source.parent / "sticker.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        img_np = _load_grayscale(source)
        img_np = _normalize_contrast(img_np)
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
        bw = _pick_threshold_candidate(img_np)

        inverted = cv2.bitwise_not(bw)
        if _thermal_polarity_score(inverted, thermal_white_target) < _thermal_polarity_score(
            bw, thermal_white_target
        ):
            bw = inverted

        bw = cv2.erode(bw, np.ones((2, 2), np.uint8), iterations=1)

        h, w = bw.shape
        if w == 0:
            raise ImageProcessingError("Sticker conversion failed: empty image width.")
        scale = target_width / w
        new_height = max(1, int(h * scale))

        bw = cv2.resize(bw, (target_width, new_height), interpolation=cv2.INTER_NEAREST)

        if float(np.mean(bw == 255)) < 0.5:
            bw = cv2.bitwise_not(bw)

        if not cv2.imwrite(str(destination), bw):
            raise ImageProcessingError(f"Could not write converted image to {destination}.")
    except ImageProcessingError:
        raise
    except Exception as exc:
        raise ImageProcessingError(f"Sticker conversion failed: {exc}") from exc

    return destination


def _png_to_monochrome_bitmap(image_path: Path, *, target_width: int) -> np.ndarray:
    img_gray = np.array(Image.open(image_path).convert("L"))
    if img_gray.size == 0:
        raise ImageProcessingError("Sticker raster conversion failed: empty image.")

    h, w = img_gray.shape
    if w == 0:
        raise ImageProcessingError("Sticker raster conversion failed: empty image width.")

    if w != target_width:
        scale = target_width / w
        new_height = max(1, int(h * scale))
        img_gray = cv2.resize(img_gray, (target_width, new_height), interpolation=cv2.INTER_NEAREST)

    _, bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    if float(np.mean(bw == 255)) < 0.5:
        bw = cv2.bitwise_not(bw)

    return bw


def convert_png_to_escpos_raster_bytes(image_path: str | Path, *, target_width: int) -> bytes:
    source = Path(image_path)
    if not source.exists():
        raise ImageProcessingError(f"Input image does not exist: {source}")

    try:
        bw = _png_to_monochrome_bitmap(source, target_width=target_width)

        # ESC/POS raster expects 1 bit per pixel, MSB first in each byte.
        black_pixels = (bw == 0).astype(np.uint8)
        width_px = black_pixels.shape[1]
        if width_px % 8 != 0:
            pad = 8 - (width_px % 8)
            black_pixels = np.pad(black_pixels, ((0, 0), (0, pad)), mode="constant", constant_values=0)

        packed = np.packbits(black_pixels, axis=1, bitorder="big")
        raster_data = packed.tobytes()

        width_bytes = packed.shape[1]
        height = black_pixels.shape[0]
        if height > 65535:
            raise ImageProcessingError(
                f"Sticker raster conversion failed: height {height} exceeds ESC/POS limit 65535."
            )

        x_l = width_bytes & 0xFF
        x_h = (width_bytes >> 8) & 0xFF
        y_l = height & 0xFF
        y_h = (height >> 8) & 0xFF

        # GS v 0 m xL xH yL yH d1...dk
        return b"\x1d\x76\x30\x00" + bytes([x_l, x_h, y_l, y_h]) + raster_data
    except ImageProcessingError:
        raise
    except Exception as exc:
        raise ImageProcessingError(f"Sticker raster conversion failed: {exc}") from exc
