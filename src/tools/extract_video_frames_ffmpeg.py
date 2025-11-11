#!/usr/bin/env python3
"""
Extract video frames using ffmpeg with optional GPU (CUDA/NVDEC) acceleration.

Features:
- Every-N-frames selection (e.g., stride=5)
- GPU decode (h264_cuvid/hevc_cuvid) when available; CPU filters for select/scale
- Parallel processing of multiple videos (--num-workers)
- Robust to partial/corrupted files (best-effort extraction)
- Generates a manifest CSV (image_path,label,video_name,frame_index,...)

Example:
  python -m src.tools.extract_video_frames_ffmpeg \
    --video-dir Deepfake-Eval-2024/video-data \
    --output-dir Deepfake-Eval-2024/video-frames \
    --metadata-csv Deepfake-Eval-2024/video-metadata-publish-with-links.csv \
    --manifest-output Deepfake-Eval-2024/video_frames_manifest.csv \
    --frame-stride 5 \
    --image-size 256 \
    --num-workers 8 \
    --hwaccel auto \
    --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


logger = logging.getLogger("extract_video_frames_ffmpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames with ffmpeg (GPU-capable)")
    p.add_argument("--video-dir", required=True, help="Directory containing video files")
    p.add_argument("--output-dir", required=True, help="Directory to write extracted frames")
    p.add_argument("--metadata-csv", required=True, help="CSV with video metadata + labels")
    p.add_argument("--manifest-output", required=True, help="Path to write output manifest CSV")
    p.add_argument("--frame-stride", type=int, default=5, help="Save one frame every N frames (default: 5)")
    p.add_argument("--image-size", type=int, default=256, help="Output frame size (square)")
    p.add_argument("--num-workers", type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument(
        "--per-video-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each ffmpeg extraction. 0 disables timeout. Default: 300",
    )
    p.add_argument(
        "--hwaccel",
        choices=["auto", "cuda", "none"],
        default="auto",
        help="Hardware acceleration: try CUDA, or force none (CPU)",
    )
    p.add_argument("--quality", type=int, default=2, help="JPEG quality factor for -q:v (lower=better). Default: 2")
    p.add_argument("--skip-existing", action="store_true", help="Skip videos whose frames already exist")
    p.add_argument("--force", action="store_true", help="Re-extract even if frames exist")
    p.add_argument("--root-prefix", default="", help="Prefix to prepend to output paths in manifest")
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata_csv(metadata_csv: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    df = pd.read_csv(metadata_csv)
    col_map = {c.lower(): c for c in df.columns}
    fname_key = None
    for key in ["filename", "video_filename", "file_path", "path", "file", "name"]:
        if key in col_map:
            fname_key = col_map[key]
            break
    if not fname_key:
        raise ValueError(f"No filename column in metadata CSV; columns={df.columns.tolist()}")
    for _, row in df.iterrows():
        name = Path(str(row[fname_key]).strip()).name
        d = row.to_dict()
        d["__lower__"] = {k.lower(): v for k, v in d.items()}
        meta[name] = d
    logger.info("Loaded metadata for %d videos", len(meta))
    return meta


def parse_label(row: Dict[str, Any]) -> int:
    lower = row.get("__lower__", {})
    for key in [
        "video ground truth",
        "ground truth",
        "ground_truth",
        "label",
        "fake",
        "is_fake",
    ]:
        if key in lower:
            v = lower[key]
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"real", "true", "0", "r"}:
                    return 0
                if s in {"fake", "false", "1", "f"}:
                    return 1
                try:
                    v = int(float(s))
                except Exception:
                    return -1
            try:
                iv = int(v)
                return iv if iv in (0, 1) else -1
            except Exception:
                return -1
    return -1


def build_ffmpeg_cmd(
    inp: Path,
    out_dir: Path,
    image_size: int,
    frame_stride: int,
    quality: int,
    hwaccel: str,
    pipeline: str,
) -> List[str]:
    """Build ffmpeg command for a specific pipeline.

    pipeline options:
      - 'gpu_decode_cpu_filters': CUDA decode (cuvid) + CPU select/scale
      - 'cpu_all': CPU decode + CPU select/scale (fallback)
    """
    # Every-N-frames selection (escape comma)
    select_expr = f"select='not(mod(n\\,{frame_stride}))'"
    vf_cpu = f"{select_expr},scale={image_size}:{image_size}:flags=bicubic"

    if pipeline == "gpu_decode_cpu_filters" and hwaccel in ("cuda", "auto"):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-hwaccel",
            "cuda",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
            # decoder will be injected by caller via ['-c:v', 'h264_cuvid'|'hevc_cuvid']
            "-i",
            str(inp),
            "-vf",
            vf_cpu,
            "-vsync",
            "vfr",
            "-q:v",
            str(quality),
            str(out_dir / "frame_%05d.jpg"),
        ]
        return cmd

    # CPU-only fallback
    cmd = [
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
        "-i",
        str(inp),
        "-vf",
        vf_cpu,
        "-vsync",
        "vfr",
        "-q:v",
        str(quality),
        str(out_dir / "frame_%05d.jpg"),
    ]
    return cmd


def try_run_ffmpeg(cmd: List[str], timeout: Optional[int] = None) -> Tuple[bool, str]:
    try:
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if timeout and timeout > 0:
            kwargs["timeout"] = timeout
        proc = subprocess.run(cmd, **kwargs)
        ok = proc.returncode == 0
        errmsg = proc.stderr.decode("utf-8", errors="ignore")
        return ok, errmsg
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError:
        return False, "ffmpeg not found in PATH"


def process_one(
    video_path: Path,
    frames_dir: Path,
    image_size: int,
    frame_stride: int,
    quality: int,
    hwaccel: str,
    per_video_timeout: int,
) -> Tuple[str, int, Optional[str], str]:
    ensure_dir(frames_dir)

    pipeline_used = ""
    ok = False
    err = ""

    # Try GPU decode (H.264)
    if hwaccel in ("cuda", "auto"):
        cmd = build_ffmpeg_cmd(
            video_path, frames_dir, image_size, frame_stride, quality, hwaccel=hwaccel, pipeline="gpu_decode_cpu_filters"
        )
        ins = cmd.index("-i")
        cmd[ins:ins] = ["-c:v", "h264_cuvid"]
        ok, err = try_run_ffmpeg(cmd, timeout=per_video_timeout)
        if ok:
            pipeline_used = "gpu_decode(h264_cuvid)+cpu_filters"

    # Try GPU decode (HEVC)
    if not ok and hwaccel in ("cuda", "auto"):
        cmd = build_ffmpeg_cmd(
            video_path, frames_dir, image_size, frame_stride, quality, hwaccel=hwaccel, pipeline="gpu_decode_cpu_filters"
        )
        ins = cmd.index("-i")
        cmd[ins:ins] = ["-c:v", "hevc_cuvid"]
        ok, err = try_run_ffmpeg(cmd, timeout=per_video_timeout)
        if ok:
            pipeline_used = "gpu_decode(hevc_cuvid)+cpu_filters"

    # CPU fallback
    if not ok:
        cmd = build_ffmpeg_cmd(
            video_path, frames_dir, image_size, frame_stride, quality, hwaccel="none", pipeline="cpu_all"
        )
        ok, err = try_run_ffmpeg(cmd, timeout=per_video_timeout)
        if ok:
            pipeline_used = "cpu"

    # Count frames
    if ok:
        count = len(list(frames_dir.glob("frame_*.jpg")))
        return (video_path.stem, count, None, pipeline_used)
    else:
        return (video_path.stem, 0, err.strip()[:500], pipeline_used or "unknown")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_output)

    ensure_dir(output_dir)
    ensure_dir(manifest_path.parent)

    if not video_dir.exists():
        logger.error("Video directory does not exist: %s", video_dir)
        return 1

    # Load metadata (labels)
    meta = load_metadata_csv(Path(args.metadata_csv))

    # Find video files
    exts = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".webm"}
    videos = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not videos:
        logger.warning("No video files found in %s", video_dir)
        return 1

    # Prepare jobs
    jobs: List[Tuple[Path, Path]] = []
    skipped: List[str] = []
    for v in videos:
        frames_dir = output_dir / f"{v.stem}_frames"
        if frames_dir.exists() and args.skip_existing and not args.force:
            skipped.append(v.stem)
        else:
            jobs.append((v, frames_dir))

    logger.info("Videos to process: %d | skipped(existing): %d", len(jobs), len(skipped))

    failures: List[Tuple[str, str]] = []
    extracted: Dict[str, int] = {}
    pipelines: Dict[str, str] = {}

    # Parallel extraction
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = {
            ex.submit(
                process_one,
                v,
                fdir,
                args.image_size,
                args.frame_stride,
                args.quality,
                args.hwaccel,
                int(max(0, args.per_video_timeout)),
            ): (v, fdir)
            for (v, fdir) in jobs
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Extracting frames"):
            stem, count, err, pipeline_used = fut.result()
            if err:
                failures.append((stem, err))
            else:
                extracted[stem] = count
                pipelines[stem] = pipeline_used

    # Build manifest rows from all frames (including skipped)
    rows: List[Dict[str, Any]] = []
    for v in videos:
        vname = v.stem
        frames_dir = output_dir / f"{vname}_frames"
        if not frames_dir.exists():
            continue

        m = meta.get(v.name)
        label = parse_label(m) if m else -1
        if label not in (0, 1):
            continue

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        for idx, f in enumerate(frame_files):
            rel = f.relative_to(output_dir.parent)
            rel_str = str(rel).replace("\\", "/")
            if args.root_prefix:
                rel_str = args.root_prefix.rstrip("/") + "/" + rel_str
            rows.append(
                {
                    "image_path": rel_str,
                    "label": label,
                    "video_name": vname,
                    "frame_index": idx,
                    "source_type": "video_frame",
                    "original_video_path": str(v.relative_to(video_dir.parent)),
                }
            )

    # Write manifest
    if rows:
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        logger.info("Saved manifest: %s (rows=%d)", manifest_path, len(rows))
    else:
        logger.warning("No rows written to manifest (no labeled frames found)")

    # Save extraction log
    log = {
        "videos_total": len(videos),
        "videos_processed": len(jobs),
        "videos_skipped": len(skipped),
        "videos_failed": len(failures),
        "extracted_summary": extracted,
        "pipelines": pipelines,
        "failures": failures,
    }
    (output_dir / "extraction_log_ffmpeg.json").write_text(json.dumps(log, indent=2))
    logger.info("Saved extraction log: %s", output_dir / "extraction_log_ffmpeg.json")

    if failures:
        logger.warning("%d videos failed during extraction. See log for details.", len(failures))

    # Print a brief pipeline summary
    if pipelines:
        gpu_cnt = sum(1 for p in pipelines.values() if p.startswith("gpu_decode"))
        cpu_cnt = sum(1 for p in pipelines.values() if p == "cpu")
        logger.info("Pipeline usage: GPU-decode=%d, CPU=%d", gpu_cnt, cpu_cnt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
