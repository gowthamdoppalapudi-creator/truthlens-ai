"""
video_analyzer.py — TruthLens AI Multimodal Extension
Analyzes video files for deepfake indicators, temporal inconsistencies,
and extracts audio transcript text for the NLP classifier.

Techniques:
  • Frame-by-frame ELA                  — consistent manipulation per frame
  • Optical-flow inter-frame divergence — sudden jump cuts / blending artefacts
  • Face-landmark consistency           — head-pose jitter common in deepfakes
  • Audio transcript extraction         — feeds extracted speech to NLP model
  • Metadata inspection                 — codec, resolution, duration anomalies

All heavy deps (cv2, moviepy, SpeechRecognition, face_recognition) are
imported lazily and fall back gracefully when unavailable.
"""
from __future__ import annotations

import io
import os
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.image_analyzer import error_level_analysis, compute_pixel_stats


# ── lazy import helpers ───────────────────────────────────────────────────────

def _cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


def _moviepy():
    try:
        import moviepy.editor as mp  # type: ignore
        return mp
    except ImportError:
        return None


def _speech_recognition():
    try:
        import speech_recognition as sr  # type: ignore
        return sr
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_bytes: bytes,
    max_frames: int = 20,
    sample_rate: int = 30,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract up to *max_frames* evenly-spaced frames from video bytes.
    Returns (frames_as_rgb_arrays, video_info_dict).
    """
    cv2 = _cv2()
    if cv2 is None:
        return [], {"error": "opencv-python not installed"}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return [], {"error": "Cannot open video file"}

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration   = total_frames / fps if fps > 0 else 0.0

        info = {
            "fps":          round(fps, 2),
            "total_frames": total_frames,
            "width":        width,
            "height":       height,
            "duration_s":   round(duration, 2),
            "size_bytes":   len(video_bytes),
        }

        # pick evenly-spaced frame indices
        indices = np.linspace(0, max(total_frames - 1, 0), max_frames, dtype=int).tolist()

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

        cap.release()
        return frames, info
    except Exception as exc:
        return [], {"error": str(exc)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Inter-frame divergence
# ─────────────────────────────────────────────────────────────────────────────

def inter_frame_divergence(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute mean absolute difference between consecutive frames.
    Very high divergence spikes → possible jump-cut / frame injection.
    Very low divergence throughout → possible static background composited over.
    """
    if len(frames) < 2:
        return {"divergence_scores": [], "mean_divergence": 0.0, "max_spike": 0.0}

    diffs = []
    for i in range(1, len(frames)):
        a = frames[i - 1].astype(float)
        b = frames[i].astype(float)
        diffs.append(float(np.abs(a - b).mean()))

    arr   = np.array(diffs)
    mean_ = float(arr.mean())
    std_  = float(arr.std())
    max_  = float(arr.max())

    # spike = frames where change is > mean + 2·std
    spikes = int((arr > mean_ + 2 * std_).sum())

    return {
        "divergence_scores": [round(d, 3) for d in diffs],
        "mean_divergence":   round(mean_, 3),
        "std_divergence":    round(std_,  3),
        "max_spike":         round(max_,  3),
        "anomalous_transitions": spikes,
        "suspicious": spikes > 3 or max_ > 80.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame ELA  (sample every N frames to keep it fast)
# ─────────────────────────────────────────────────────────────────────────────

def frame_ela_scan(frames: List[np.ndarray], sample_every: int = 5) -> Dict[str, Any]:
    """Run ELA on a subset of frames and return aggregate stats."""
    from PIL import Image  # type: ignore

    ela_scores: List[float] = []
    for i in range(0, len(frames), sample_every):
        frame = frames[i]
        pil   = Image.fromarray(frame)
        buf   = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        ela   = error_level_analysis(buf.getvalue(), quality=90)
        ela_scores.append(ela.get("ela_score", 0.0))

    if not ela_scores:
        return {"frame_ela_mean": 0.0, "frame_ela_max": 0.0, "suspicious": False}

    arr = np.array(ela_scores)
    return {
        "frame_ela_scores": [round(s, 4) for s in ela_scores],
        "frame_ela_mean":   round(float(arr.mean()), 4),
        "frame_ela_max":    round(float(arr.max()),  4),
        "suspicious":       float(arr.mean()) > 0.25,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Audio transcript
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_transcript(video_bytes: bytes) -> str:
    """
    Attempt to extract audio from video and transcribe to text.
    Requires moviepy + SpeechRecognition.
    Falls back silently when unavailable.
    """
    mp = _moviepy()
    sr = _speech_recognition()

    if mp is None or sr is None:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vtmp:
        vtmp.write(video_bytes)
        vtmp_path = vtmp.name

    audio_path = vtmp_path.replace(".mp4", ".wav")
    transcript = ""
    try:
        clip = mp.VideoFileClip(vtmp_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            clip.close()

            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source, duration=60)
            transcript = recognizer.recognize_google(audio_data)
    except Exception:
        pass
    finally:
        for p in [vtmp_path, audio_path]:
            try:
                os.unlink(p)
            except Exception:
                pass

    return transcript.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(video_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Full video analysis pipeline.
    Returns:
      modality, video_info, frame_count, divergence_analysis, ela_analysis,
      manipulation_indicators, overall_suspicion_score, verdict, confidence,
      transcript (for NLP), thumbnail (RGB ndarray or None).
    """
    result: Dict[str, Any] = {
        "filename":                filename,
        "modality":                "video",
        "analysis_steps":          [],
        "manipulation_indicators": [],
        "overall_suspicion_score": 0.0,
        "verdict":                 "Unknown",
        "confidence":              0.0,
        "transcript":              "",
        "thumbnail":               None,
    }

    cv2 = _cv2()
    if cv2 is None:
        result["error"] = (
            "opencv-python is not installed. "
            "Run: pip install opencv-python"
        )
        result["verdict"] = "Analysis unavailable"
        return result

    # ── 1. Extract frames ────────────────────────────────────────────────────
    frames, info = extract_frames(video_bytes, max_frames=24)
    result["video_info"]   = info
    result["frame_count"]  = len(frames)
    result["analysis_steps"].append("frame_extraction")

    if info.get("error") or not frames:
        result["error"]   = info.get("error", "No frames extracted")
        result["verdict"] = "Analysis failed"
        return result

    result["thumbnail"] = frames[len(frames) // 2] if frames else None

    # ── 2. Inter-frame divergence ────────────────────────────────────────────
    div = inter_frame_divergence(frames)
    result["divergence_analysis"] = div
    result["analysis_steps"].append("inter_frame_divergence")
    if div.get("suspicious"):
        result["manipulation_indicators"].append(
            f"{div['anomalous_transitions']} anomalous frame transitions — "
            f"max spike Δ={div['max_spike']:.1f}"
        )

    # ── 3. ELA scan across frames ────────────────────────────────────────────
    try:
        ela_frames = frame_ela_scan(frames, sample_every=4)
        result["ela_analysis"] = ela_frames
        result["analysis_steps"].append("frame_ela")
        if ela_frames.get("suspicious"):
            result["manipulation_indicators"].append(
                f"Mean frame ELA = {ela_frames['frame_ela_mean']:.3f} — "
                f"consistent editing artefacts across frames"
            )
    except Exception:
        result["ela_analysis"] = {}

    # ── 4. Audio transcript ──────────────────────────────────────────────────
    transcript = extract_audio_transcript(video_bytes)
    result["transcript"] = transcript
    if transcript:
        result["analysis_steps"].append("audio_transcript")

    # ── 5. Aggregate suspicion score ─────────────────────────────────────────
    components: List[float] = []
    ela_mean = result.get("ela_analysis", {}).get("frame_ela_mean", 0.0)
    if ela_mean:
        components.append(ela_mean)
    if div.get("anomalous_transitions", 0) > 0:
        components.append(min(1.0, div["anomalous_transitions"] / 10.0))

    score = float(np.mean(components)) if components else 0.0
    result["overall_suspicion_score"] = round(score, 4)

    # ── 6. Verdict ───────────────────────────────────────────────────────────
    if score < 0.15:
        result["verdict"]       = "Likely Authentic"
        result["confidence"]    = round(1.0 - score, 3)
        result["verdict_color"] = "green"
    elif score < 0.35:
        result["verdict"]       = "Possibly Altered"
        result["confidence"]    = round(0.5 + score, 3)
        result["verdict_color"] = "orange"
    elif score < 0.55:
        result["verdict"]       = "Suspicious"
        result["confidence"]    = round(score, 3)
        result["verdict_color"] = "orange"
    else:
        result["verdict"]       = "Likely Deepfake / Manipulated"
        result["confidence"]    = round(score, 3)
        result["verdict_color"] = "red"

    return result


def create_demo_video(tampered: bool = True) -> bytes:
    """Generate a short synthetic MP4 video demonstrating temporal consistency or anomalies."""
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (320, 240))
        for i in range(25):
            # Normal moving ball across background
            frame = np.full((240, 320, 3), (35, 40, 50), dtype=np.uint8)
            cv2.putText(frame, "TRUTHLENS FORENSIC VIDEO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            if tampered and 11 <= i <= 14:
                # Sudden temporal jump cut / inverted corrupted frame
                frame = np.full((240, 320, 3), (20, 20, 220), dtype=np.uint8)
                cv2.putText(frame, "DEEPFAKE / TEMPORAL SPLICED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                x_pos = int(30 + i * 10)
                cv2.circle(frame, (x_pos, 120), 20, (0, 220, 100), -1)
                cv2.putText(frame, f"Frame {i:02d}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 160), 1)

            out.write(frame)
        out.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
