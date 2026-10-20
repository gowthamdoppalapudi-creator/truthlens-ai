"""
image_analyzer.py — TruthLens AI Multimodal Extension
Analyzes images for authenticity, manipulation, and fake-news indicators.

Techniques:
  • Error Level Analysis (ELA)  — detects JPEG re-compression artefacts
  • EXIF metadata inspection    — flags stripped/modified timestamps
  • Pixel statistics             — colour channel uniformity checks
  • Gradient / noise analysis    — copy-paste region detection heuristic
  • Text extraction via OCR      — feeds extracted text to the NLP classifier
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── optional deps (graceful fallback) ────────────────────────────────────────

def _pil() -> Any:
    try:
        from PIL import Image, ImageChops, ImageEnhance, ExifTags
        return Image, ImageChops, ImageEnhance, ExifTags
    except ImportError:
        return None

def _tesseract_available() -> bool:
    try:
        import pytesseract  # type: ignore
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ELA — Error Level Analysis
# ─────────────────────────────────────────────────────────────────────────────

def error_level_analysis(
    image_bytes: bytes,
    quality: int = 90,
) -> Dict[str, Any]:
    """
    Perform Error Level Analysis.
    Re-saves the image at *quality* and measures pixel-wise differences.
    High ELA in 'flat' regions (sky, solid backgrounds) indicates compositing.
    Returns ela_image (PIL.Image or None) plus numeric scores.
    """
    mods = _pil()
    if mods is None:
        return {"error": "Pillow not installed", "ela_score": 0.0, "suspicious": False}

    Image, ImageChops, ImageEnhance, _ = mods
    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        diff = ImageChops.difference(original, recompressed)
        arr = np.array(diff, dtype=np.float32)

        mean_err = float(arr.mean())
        max_err  = float(arr.max())
        std_err  = float(arr.std())

        # scale factor so a "clean" image → ~0 and a heavily-edited one → ~1
        ela_score = min(1.0, mean_err / 20.0)

        # enhanced version for display (scale ×15)
        enhancer  = ImageEnhance.Brightness(diff)
        ela_image = enhancer.enhance(15)

        return {
            "ela_score":  round(ela_score, 4),
            "mean_error": round(mean_err, 3),
            "max_error":  round(max_err,  3),
            "std_error":  round(std_err,  3),
            "ela_image":  ela_image,
            "suspicious": ela_score > 0.30,
        }
    except Exception as exc:
        return {"error": str(exc), "ela_score": 0.0, "suspicious": False}


# ─────────────────────────────────────────────────────────────────────────────
# EXIF / Metadata
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata(image_bytes: bytes) -> Dict[str, Any]:
    """Extract image dimensions, format, and EXIF fields."""
    mods = _pil()
    if mods is None:
        return {"error": "Pillow not installed"}

    Image, _, __, ExifTags = mods
    try:
        img = Image.open(io.BytesIO(image_bytes))
        meta: Dict[str, Any] = {
            "format":     img.format or "unknown",
            "mode":       img.mode,
            "width":      img.width,
            "height":     img.height,
            "size_bytes": len(image_bytes),
            "megapixels": round((img.width * img.height) / 1_000_000, 2),
        }

        exif: Dict[str, Any] = {}
        try:
            raw = img._getexif()  # type: ignore[attr-defined]
            if raw:
                for tag_id, val in raw.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8", errors="replace")
                        except Exception:
                            val = repr(val)
                    if isinstance(val, (str, int, float)):
                        exif[str(tag)] = val
        except Exception:
            pass

        meta["exif"]     = exif
        meta["has_exif"] = bool(exif)

        # suspicious patterns
        signals: List[str] = []
        if not exif:
            signals.append("EXIF data absent — may have been deliberately stripped")
        software = exif.get("Software", "")
        if software:
            signals.append(f"Edited with software: {software}")
        dt_orig = exif.get("DateTimeOriginal")
        dt_mod  = exif.get("DateTime")
        if dt_orig and dt_mod and dt_orig != dt_mod:
            signals.append(
                f"Modification date ({dt_mod}) differs from capture date ({dt_orig})"
            )
        if exif.get("Flash") == 0 and exif.get("MeteringMode") in (None,):
            pass  # benign

        meta["suspicious_signals"] = signals
        return meta
    except Exception as exc:
        return {"error": str(exc), "suspicious_signals": []}


# ─────────────────────────────────────────────────────────────────────────────
# Pixel Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_pixel_stats(image_bytes: bytes) -> Dict[str, Any]:
    """Channel-level stats + gradient-based noise estimate."""
    mods = _pil()
    if mods is None:
        return {}

    Image = mods[0]
    try:
        img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr  = np.array(img, dtype=np.float32)

        stats: Dict[str, Any] = {
            "aspect_ratio": round(img.width / max(img.height, 1), 4),
        }
        for i, ch in enumerate(["red", "green", "blue"]):
            c = arr[:, :, i]
            stats[ch] = {
                "mean": round(float(c.mean()), 2),
                "std":  round(float(c.std()),  2),
                "min":  round(float(c.min()),  2),
                "max":  round(float(c.max()),  2),
            }

        # Simple Laplacian noise via finite differences (no scipy needed)
        gray = arr.mean(axis=2)
        dx   = np.diff(gray, axis=1)
        dy   = np.diff(gray, axis=0)
        noise_var = float(dx.var() + dy.var())
        stats["noise_variance"]  = round(noise_var, 2)
        stats["sharpness_score"] = round(min(1.0, noise_var / 2000.0), 4)

        # Look for abnormally uniform patches (pasted regions)
        block = 32
        h, w  = gray.shape
        block_vars: List[float] = []
        for r in range(0, h - block, block):
            for c in range(0, w - block, block):
                block_vars.append(float(gray[r:r+block, c:c+block].var()))
        if block_vars:
            bv_arr = np.array(block_vars)
            stats["block_variance_mean"] = round(float(bv_arr.mean()), 2)
            stats["block_variance_std"]  = round(float(bv_arr.std()),  2)
            # many near-zero blocks ⇒ suspicious pasting
            zero_blocks = int((bv_arr < 1.0).sum())
            stats["near_zero_variance_blocks"] = zero_blocks
            stats["pasting_suspicion"] = zero_blocks / max(len(block_vars), 1) > 0.10

        return stats
    except Exception as exc:
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# OCR — Extract embedded text
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Run Tesseract OCR on the image and return extracted text.
    Falls back silently if pytesseract is not installed.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        return pytesseract.image_to_string(img).strip()
    except ImportError:
        return ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Full image-authenticity analysis.
    Returns a structured dict with:
      modality, metadata, ela_analysis, pixel_stats,
      manipulation_indicators, overall_suspicion_score, verdict, confidence,
      extracted_text (for NLP pipeline), ela_image (PIL Image or None).
    """
    result: Dict[str, Any] = {
        "filename":               filename,
        "modality":               "image",
        "analysis_steps":         [],
        "manipulation_indicators": [],
        "overall_suspicion_score": 0.0,
        "verdict":                "Unknown",
        "confidence":             0.0,
        "extracted_text":         "",
        "ela_image":              None,
    }

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    meta = extract_metadata(image_bytes)
    result["metadata"] = meta
    result["analysis_steps"].append("metadata")
    result["manipulation_indicators"].extend(meta.get("suspicious_signals", []))

    # ── 2. ELA ───────────────────────────────────────────────────────────────
    ela = error_level_analysis(image_bytes)
    result["ela_analysis"] = {k: v for k, v in ela.items() if k != "ela_image"}
    result["ela_image"]    = ela.get("ela_image")
    result["analysis_steps"].append("ela")
    if ela.get("suspicious"):
        result["manipulation_indicators"].append(
            f"ELA score {ela.get('ela_score', 0):.3f} — possible compositing / re-editing"
        )

    # ── 3. Pixel stats ───────────────────────────────────────────────────────
    pix = compute_pixel_stats(image_bytes)
    result["pixel_stats"] = pix
    result["analysis_steps"].append("pixel_stats")
    if pix.get("pasting_suspicion"):
        result["manipulation_indicators"].append(
            "Suspiciously uniform pixel blocks detected — possible copy-paste"
        )

    # ── 4. OCR ───────────────────────────────────────────────────────────────
    ocr_text = extract_text_from_image(image_bytes)
    result["extracted_text"] = ocr_text
    if ocr_text:
        result["analysis_steps"].append("ocr")

    # ── 5. Compute aggregate suspicion score ─────────────────────────────────
    components: List[float] = []
    if ela.get("ela_score", 0) > 0:
        components.append(ela["ela_score"])
    n_signals = len(meta.get("suspicious_signals", []))
    if n_signals:
        components.append(min(1.0, n_signals * 0.25))
    if pix.get("pasting_suspicion"):
        components.append(0.5)

    score = float(np.mean(components)) if components else 0.0
    result["overall_suspicion_score"] = round(score, 4)

    # ── 6. Verdict ───────────────────────────────────────────────────────────
    if score < 0.15:
        result["verdict"]     = "Likely Authentic"
        result["confidence"]  = round(1.0 - score, 3)
        result["verdict_color"] = "green"
    elif score < 0.35:
        result["verdict"]     = "Possibly Altered"
        result["confidence"]  = round(0.5 + score, 3)
        result["verdict_color"] = "orange"
    elif score < 0.55:
        result["verdict"]     = "Suspicious"
        result["confidence"]  = round(score, 3)
        result["verdict_color"] = "orange"
    else:
        result["verdict"]     = "Likely Manipulated"
        result["confidence"]  = round(score, 3)
        result["verdict_color"] = "red"

    return result


def create_demo_image(tampered: bool = False) -> bytes:
    """Generate a synthetic test image with natural or tampered characteristics."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 400), color=(240, 244, 248))
    draw = ImageDraw.Draw(img)

    # Draw header banner
    draw.rectangle([0, 0, 640, 90], fill=(22, 27, 34))
    draw.text((30, 32), "TRUTHLENS AI FORENSIC RESEARCH LAB — PRESS RELEASE", fill=(255, 255, 255))

    if tampered:
        # Authentic baseline text
        draw.text((30, 120), "OFFICIAL GOVERNMENT BULLETIN:", fill=(60, 64, 72))
        draw.text((30, 155), "Municipal transit systems report standard operating schedule.", fill=(80, 84, 92))

        # Spliced altered patch with different noise and compression
        patch = Image.new("RGB", (560, 85), color=(255, 225, 225))
        p_draw = ImageDraw.Draw(patch)
        p_draw.text((20, 28), "URGENT: ALIEN HYPNOTIC NANOBOTS DETECTED IN CITY DRINKING WATER!", fill=(200, 20, 20))
        buf_patch = io.BytesIO()
        patch.save(buf_patch, format="JPEG", quality=25)
        buf_patch.seek(0)
        patch_recomp = Image.open(buf_patch)
        img.paste(patch_recomp, (30, 220))
    else:
        draw.text((30, 120), "OFFICIAL SCIENTIFIC BRIEFING:", fill=(60, 64, 72))
        draw.text((30, 155), "NASA astronomers confirm atmospheric water vapor on exoplanet K2-18b.", fill=(40, 44, 52))
        draw.text((30, 190), "Peer-reviewed spectroscopic data acquired via James Webb Space Telescope.", fill=(60, 64, 72))
        draw.rectangle([30, 245, 610, 340], fill=(225, 238, 250), outline=(160, 195, 235))
        draw.text((50, 280), "Archival Reference: International Astronomical Consortium Document #842", fill=(45, 75, 120))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
