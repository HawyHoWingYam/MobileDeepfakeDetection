#!/usr/bin/env python3
"""
Extract frames from videos in HuggingFace Deepfake-Eval-2024 dataset.

GPU-first implementation: uses FFmpeg with CUDA/NVDEC for hardware-accelerated
decode when available, falling back to CPU if necessary. OpenCV CPU extraction
is used only as a last resort. This keeps the tool robust on machines without
GPU-enabled FFmpeg while ensuring GPU is used when present.

This tool:
1. Loads video files from local directory (recursive)
2. Prefers FFmpeg GPU decode (h264_cuvid/hevc_cuvid) with frame stride or FPS
3. Saves frames as JPEG images with organized directory structure
4. Loads metadata CSV to retrieve labels and other info
5. Generates a manifest CSV compatible with project dataset format

Example:
    python -m src.tools.extract_video_frames_hf \
        --video-dir dataset/deepfake_eval_2024/raw/video-data \
        --output-dir dataset/deepfake_eval_2024/video_frames \
        --metadata-csv dataset/deepfake_eval_2024/raw/video-metadata-publish-with-links.csv \
        --manifest-output dataset/deepfake_eval_2024/video_frames_manifest.csv \
        --fps 1 \
        --frame-stride 5 \
        --max-frames-per-video 30 \
        --image-size 256 \
        --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("extract_video_frames_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from video files and create manifest CSV"
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store extracted frames",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="CSV file with video metadata (filename, label, etc.)",
    )
    parser.add_argument(
        "--manifest-output",
        required=True,
        help="Path to save output manifest CSV",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help=(
            "Target extraction FPS (frames per second). For example, 1.0 extracts ~1 frame per second. "
            "If --frame-stride is provided, it takes precedence."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "Extract 1 frame every N frames (overrides --fps). For example, 5 saves every 5th frame."
        ),
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=30,
        help="Maximum number of frames to extract per video",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize extracted frames to this size (square)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have frames extracted",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract frames even if they exist (overrides --skip-existing)",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Prefix to prepend to relative frame paths in manifest",
    )
    parser.add_argument(
        "--hwaccel",
        choices=["auto", "cuda", "none"],
        default="auto",
        help=(
            "Hardware acceleration preference for FFmpeg: 'cuda' to force NVDEC,\n"
            "'auto' to try CUDA then CPU, or 'none' for CPU only."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=2,
        help="JPEG quality for FFmpeg '-q:v' (lower is better; default=2)",
    )
    parser.add_argument(
        "--ffmpeg-timeout",
        type=int,
        default=300,
        help=(
            "Timeout in seconds per FFmpeg invocation. If exceeded, the current video "
            "is aborted. When '--hwaccel auto', the tool retries once on CPU."
        ),
    )
    parser.add_argument(
        "--retry-with-cpu-on-timeout",
        action="store_true",
        help=(
            "If FFmpeg with GPU times out and '--hwaccel' is 'auto', retry the same video on CPU."
        ),
    )
    parser.add_argument(
        "--force-opencv",
        action="store_true",
        help=(
            "Skip FFmpeg entirely and use OpenCV+CPU extraction (more stable, slower)."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata_csv(metadata_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load metadata CSV and index by filename (case-insensitive headers).

    Supports Deepfake‑Eval‑2024 video metadata which uses 'Filename' and
    'Video Ground Truth' headers, as well as more generic schemas.
    """
    metadata: Dict[str, Dict[str, Any]] = {}
    if not metadata_csv.exists():
        logger.warning("Metadata CSV not found: %s", metadata_csv)
        return metadata

    try:
        df = pd.read_csv(metadata_csv)

        # Normalize columns for robust access
        col_map = {c.lower(): c for c in df.columns}

        # Try common filename column names (case-insensitive)
        filename_key = None
        for key in ["filename", "video_filename", "file_path", "path", "file", "name"]:
            if key in col_map:
                filename_key = col_map[key]
                break

        if not filename_key:
            logger.warning(
                "No recognized filename column in metadata CSV. Columns: %s",
                df.columns.tolist(),
            )
            return metadata

        for _, row in df.iterrows():
            fname = str(row[filename_key]).strip()
            fname_base = Path(fname).name

            # Also attach a lowercase-key copy for easier downstream parsing
            row_dict = row.to_dict()
            row_lower = {k.lower(): v for k, v in row_dict.items()}
            row_dict["__lower__"] = row_lower
            metadata[fname_base] = row_dict

        logger.info("Loaded metadata for %d videos", len(metadata))
    except Exception as e:
        logger.error("Error loading metadata CSV: %s", e)

    return metadata


def _parse_label_from_metadata(meta: Dict[str, Any]) -> int:
    """Parse label from heterogeneous metadata rows.

    Returns 0 for real, 1 for fake, -1 if unknown/unparsable.
    """
    # Prefer lowercase alias map if present
    row_lower = meta.get("__lower__", {})

    # Candidate label fields in priority order
    candidates = [
        ("label", None),
        ("fake", None),
        ("ground truth", None),
        ("video ground truth", None),
        ("video_ground_truth", None),
        ("ground_truth", None),
        ("is_fake", None),
    ]

    value: Any = None
    for key, _ in candidates:
        if key in row_lower:
            value = row_lower.get(key)
            break
        if key in meta:
            value = meta.get(key)
            break

    if value is None:
        return -1

    # Normalize textual labels
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"real", "true", "0", "r"}:
            return 0
        if v in {"fake", "false", "1", "f"}:
            return 1
        # Some CSVs might use 'Unknown'
        if v in {"unknown", "na", "n/a", ""}:
            return -1
        # Try numeric cast
        try:
            value = int(float(v))
        except Exception:
            return -1

    # Numeric-like
    try:
        iv = int(value)
        if iv in (0, 1):
            return iv
    except Exception:
        pass

    return -1


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _try_run(cmd: List[str], timeout: Optional[int] = None) -> Tuple[bool, str, bool]:
    """Run a command, returning (ok, stderr_text, timed_out)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stderr.decode("utf-8", errors="ignore"), False
    except subprocess.TimeoutExpired as e:
        return False, str(e), True
    except FileNotFoundError:
        return False, "ffmpeg not found in PATH", False


def _build_ffmpeg_cmd(
    inp: Path,
    out_dir: Path,
    image_size: int,
    target_fps: float,
    frame_stride: Optional[int],
    quality: int,
    hwaccel: str,
    cuda_decoder: Optional[str],
) -> List[str]:
    # Build filter: prefer stride selection, else fps
    if frame_stride is not None and frame_stride > 0:
        select_expr = f"select='not(mod(n\\,{int(frame_stride)}))'"
        vf = f"{select_expr},scale={image_size}:{image_size}:flags=bicubic"
    else:
        fps_val = max(1e-6, float(target_fps))
        vf = f"fps={fps_val},scale={image_size}:{image_size}:flags=bicubic"

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
    ]

    # Hardware accel
    if hwaccel in ("cuda", "auto") and cuda_decoder:
        cmd += ["-hwaccel", "cuda", "-c:v", cuda_decoder]

    cmd += [
        "-i",
        str(inp),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-q:v",
        str(int(quality)),
        str(out_dir / "frame_%05d.jpg"),
    ]
    return cmd


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 1.0,
    max_frames: int = 30,
    image_size: int = 256,
    frame_stride: Optional[int] = None,
    hwaccel: str = "auto",
    jpeg_quality: int = 2,
    ffmpeg_timeout: Optional[int] = 300,
    retry_with_cpu_on_timeout: bool = False,
    force_opencv: bool = False,
) -> Tuple[list[Path], Dict[str, Any]]:
    """Extract frames from a video file, preferring GPU decode via FFmpeg.

    Returns: (list of extracted frame paths, stats dict)
    """
    stats: Dict[str, Any] = {
        "video_path": str(video_path),
        "frames_extracted": 0,
        "total_frames": 0,
        "video_fps": 0.0,
        "duration_sec": 0.0,
        "error": None,
        "pipeline": "",
    }

    ensure_dir(output_dir)

    # If GPU is requested but ffmpeg is unavailable, do not fallback to CPU (unless force_opencv)
    if hwaccel == "cuda" and not _ffmpeg_available() and not force_opencv:
        stats["pipeline"] = "gpu_requested_but_ffmpeg_missing"
        stats["error"] = "FFmpeg not found; cannot use CUDA/NVDEC."
        return [], stats

    # Preferred path: FFmpeg with GPU decode
    used_ffmpeg = False
    if (not force_opencv) and _ffmpeg_available():
        # Try CUDA/NVDEC (h264 -> hevc), then CPU
        tried = []
        for decoder in (["h264_cuvid", "hevc_cuvid"] if hwaccel in ("cuda", "auto") else [None]):
            cmd = _build_ffmpeg_cmd(
                video_path,
                output_dir,
                image_size,
                target_fps,
                frame_stride,
                jpeg_quality,
                hwaccel,
                decoder,
            )
            ok, err, timed_out = _try_run(cmd, timeout=ffmpeg_timeout)
            tried.append(("gpu:" + (decoder or "none"), ok, err, timed_out))
            if ok:
                used_ffmpeg = True
                stats["pipeline"] = f"ffmpeg+{decoder or 'cpu'}"
                break
            if timed_out:
                logger.warning(
                    "FFmpeg timed out (%ss) for %s with %s; %s",
                    ffmpeg_timeout,
                    video_path.name,
                    decoder or "cpu",
                    "will retry on CPU" if (hwaccel in ("auto", "cuda") and retry_with_cpu_on_timeout) else "skipping GPU",
                )
                # If GPU path times out and retry is allowed later, we'll fall through to CPU

        # CPU-only fallback with FFmpeg (allowed only when not forcing GPU)
        if not used_ffmpeg and hwaccel in ("auto", "none"):
            cmd = _build_ffmpeg_cmd(
                video_path,
                output_dir,
                image_size,
                target_fps,
                frame_stride,
                jpeg_quality,
                hwaccel="none",
                cuda_decoder=None,
            )
            ok, err, timed_out = _try_run(cmd, timeout=ffmpeg_timeout if retry_with_cpu_on_timeout else None)
            tried.append(("cpu", ok, err, timed_out))
            if ok:
                used_ffmpeg = True
                stats["pipeline"] = "ffmpeg+cpu"

        if not used_ffmpeg:
            # Log the last error for visibility and fall back
            last_err = tried[-1][2] if tried else "FFmpeg failed"
            logger.warning("FFmpeg extraction failed for %s: %s", video_path.name, last_err)
            # If GPU was explicitly requested, do not fallback to CPU
            if hwaccel == "cuda":
                stats["pipeline"] = "ffmpeg+gpu_failed"
                stats["error"] = last_err or "GPU decode failed"
                return [], stats

    # If FFmpeg succeeded, enumerate frames and return
    if used_ffmpeg:
        frames = sorted(output_dir.glob("frame_*.jpg"))
        if max_frames is not None and len(frames) > max_frames:
            frames = frames[:max_frames]
        stats["frames_extracted"] = len(frames)
        return frames, stats

    # Last resort: OpenCV CPU path (or when --force-opencv)
    try:
        if hwaccel == "cuda" and not force_opencv:
            # Respect user's request to avoid CPU fallback
            stats["pipeline"] = "opencv+cpu_skipped_due_to_gpu_request"
            stats["error"] = "GPU requested; skipping CPU fallback"
            return [], stats
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            stats["error"] = "Failed to open video"
            return [], stats

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats["video_fps"] = float(video_fps)
        stats["total_frames"] = total_frames
        stats["duration_sec"] = float(total_frames / video_fps) if video_fps and video_fps > 0 else 0

        # Determine frame interval
        if frame_stride is not None and frame_stride > 0:
            frame_interval = int(frame_stride)
        else:
            if not video_fps or video_fps <= 0:
                stats["error"] = f"Invalid video FPS: {video_fps}"
                cap.release()
                return [], stats
            frame_interval = max(1, int(round(video_fps / max(1e-6, target_fps))))

        extracted_frames: List[Path] = []
        frame_idx = 0
        extracted_count = 0

        while max_frames is None or extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                out_path = output_dir / f"frame_{extracted_count:05d}.jpg"
                cv2.imwrite(str(out_path), frame_resized)
                extracted_frames.append(out_path)
                extracted_count += 1
            frame_idx += 1

        cap.release()
        stats["frames_extracted"] = extracted_count
        stats["pipeline"] = "opencv+cpu"
        return extracted_frames, stats

    except Exception as e:
        logger.error("Error extracting frames from %s: %s", video_path, e)
        stats["error"] = str(e)
        return [], stats


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_output)

    if not video_dir.exists():
        logger.error("Video directory does not exist: %s", video_dir)
        return 1

    ensure_dir(output_dir)
    ensure_dir(manifest_path.parent)

    # Load metadata if provided
    metadata_dict = {}
    if args.metadata_csv:
        metadata_dict = load_metadata_csv(Path(args.metadata_csv))

    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    video_files = sorted(
        [f for f in video_dir.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions]
    )

    if not video_files:
        logger.warning("No video files found in %s", video_dir)
        return 1

    logger.info("Found %d video files to process", len(video_files))

    manifest_rows = []
    skipped_videos: list[str] = []
    failed_videos: list[tuple[str, str]] = []
    pipelines: dict[str, str] = {}

    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = video_path.stem
        frames_output_dir = output_dir / f"{video_name}_frames"

        # Check if already extracted
        if frames_output_dir.exists() and args.skip_existing and not args.force:
            logger.debug("Skipping existing frames for video: %s", video_name)
            skipped_videos.append(video_name)
            pipelines[video_name] = "skipped"
            # Still load metadata for manifest
            metadata = metadata_dict.get(video_path.name, {})
            label = _parse_label_from_metadata(metadata) if metadata else -1

            # Add manifest rows for existing frames
            frame_files = sorted(frames_output_dir.glob("frame_*.jpg"))
            for frame_idx, frame_path in enumerate(frame_files):
                rel_path = frame_path.relative_to(output_dir.parent)
                if args.root_prefix:
                    rel_path_str = args.root_prefix.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
                else:
                    rel_path_str = str(rel_path).replace("\\", "/")

                if label in (0, 1):  # only include labeled samples
                    manifest_rows.append(
                        {
                            "image_path": rel_path_str,
                            "label": label,
                            "video_name": video_name,
                            "frame_index": frame_idx,
                            "source_type": "video_frame",
                        }
                    )
            continue

        # Extract frames
        ensure_dir(frames_output_dir)
        frame_paths, stats = extract_frames(
            video_path,
            frames_output_dir,
            target_fps=args.fps,
            max_frames=args.max_frames_per_video,
            image_size=args.image_size,
            frame_stride=args.frame_stride,
            hwaccel=args.hwaccel,
            jpeg_quality=args.jpeg_quality,
            ffmpeg_timeout=args.ffmpeg_timeout,
            retry_with_cpu_on_timeout=args.retry_with_cpu_on_timeout,
            force_opencv=args.force_opencv,
        )

        pipelines[video_name] = stats.get("pipeline", "")

        if stats["error"]:
            logger.warning("Failed to extract frames from %s: %s", video_name, stats["error"])
            failed_videos.append((video_name, stats["error"]))
            continue

        logger.debug("Extracted %d frames from %s", stats["frames_extracted"], video_name)

        # Get label from metadata
        metadata = metadata_dict.get(video_path.name, {})
        label = _parse_label_from_metadata(metadata) if metadata else -1
        if label not in (0, 1):
            logger.warning("Skipping unlabeled/unknown video %s (label=%s)", video_name, label)

        # Add manifest rows for extracted frames
        for frame_idx, frame_path in enumerate(frame_paths):
            rel_path = frame_path.relative_to(output_dir.parent)
            if args.root_prefix:
                rel_path_str = args.root_prefix.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
            else:
                rel_path_str = str(rel_path).replace("\\", "/")

            if label in (0, 1):
                manifest_rows.append(
                    {
                        "image_path": rel_path_str,
                        "label": label,
                        "video_name": video_name,
                        "frame_index": frame_idx,
                        "source_type": "video_frame",
                        "fps": stats["video_fps"],
                        "original_video_path": str(video_path.relative_to(video_dir.parent)),
                    }
                )

    # Save manifest
    if manifest_rows:
        fieldnames = sorted(set().union(*[row.keys() for row in manifest_rows]))
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest_rows:
                # Fill missing fields with empty string
                filled_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(filled_row)
        logger.info("Manifest saved: %s (%d frames from %d videos)", manifest_path, len(manifest_rows), len(video_files))
    else:
        logger.warning("No frames were extracted")

    # Save extraction log
    # Derive GPU/CPU usage summary
    gpu_decode_count = sum(
        1
        for p in pipelines.values()
        if isinstance(p, str) and (p.startswith("ffmpeg+h264_cuvid") or p.startswith("ffmpeg+hevc_cuvid"))
    )
    cpu_ffmpeg_count = sum(1 for p in pipelines.values() if p == "ffmpeg+cpu")
    opencv_cpu_count = sum(1 for p in pipelines.values() if p == "opencv+cpu")
    gpu_failed_count = sum(
        1
        for p in pipelines.values()
        if p in {"ffmpeg+gpu_failed", "gpu_requested_but_ffmpeg_missing", "opencv+cpu_skipped_due_to_gpu_request"}
    )
    skipped_count = sum(1 for p in pipelines.values() if p == "skipped")

    log_summary = {
        "total_videos": len(video_files),
        "skipped_videos": len(skipped_videos),
        "failed_videos": len(failed_videos),
        "total_frames_extracted": len(manifest_rows),
        "failed_details": [(v, e) for v, e in failed_videos],
        "pipelines": pipelines,
        "hwaccel": args.hwaccel,
        "gpu_decode_count": gpu_decode_count,
        "cpu_ffmpeg_count": cpu_ffmpeg_count,
        "opencv_cpu_count": opencv_cpu_count,
        "gpu_failed_count": gpu_failed_count,
        "skipped_count": skipped_count,
    }

    log_path = output_dir / "extraction_log.json"
    with log_path.open("w") as f:
        json.dump(log_summary, f, indent=2)
    logger.info("Extraction log saved: %s", log_path)

    # Console summary of GPU usage
    logger.info(
        "GPU decode videos: %d | CPU(ffmpeg): %d | OpenCV CPU: %d | GPU failed: %d | skipped: %d",
        log_summary["gpu_decode_count"],
        log_summary["cpu_ffmpeg_count"],
        log_summary["opencv_cpu_count"],
        log_summary["gpu_failed_count"],
        log_summary["skipped_count"],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
