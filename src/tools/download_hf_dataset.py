#!/usr/bin/env python3
"""
Download a HuggingFace dataset split, export images, and build a manifest CSV.

Example:
    python -m src.tools.download_hf_dataset \
        --dataset-id nuriachandra/Deepfake-Eval-2024 \
        --split train \
        --output-dir data/deepfake_eval_2024/train
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import re  # NEW: shard filename matching
import io  # NEW: bytes buffer for URL fetch

try:
    from datasets import Image, load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The 'datasets' package is required. Install with `pip install datasets`."
    ) from exc

# Optional dependency for restricted-shard mode
try:  # NEW: import huggingface_hub lazily
    from huggingface_hub import (
        list_repo_files,
        hf_hub_download,
        snapshot_download,
        login,
    )  # type: ignore
except Exception:  # pragma: no cover
    list_repo_files = None
    hf_hub_download = None
    snapshot_download = None
    login = None
try:  # NEW: for URL downloads when CSV provides image_url
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


logger = logging.getLogger("download_hf_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset split and export images + manifest."
    )
    parser.add_argument("--dataset-id", required=True, help="Dataset repository id (e.g., user/dataset)")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    parser.add_argument(
        "--config",
        default=None,
        help="Dataset configuration name (if the dataset exposes multiple configs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store images and manifest.csv",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column name containing the image data",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Column name containing label values",
    )
    parser.add_argument(
        "--label-map",
        default=None,
        help="Comma-separated label mapping (e.g., 'audio-data=0,video-data=1' or 'real=0,fake=1'). If not provided, uses sequential mapping.",
    )
    parser.add_argument(
        "--root-prefix",
        default="",
        help="Prefix to prepend to relative image paths in manifest (e.g., 'dataset/deepfake_eval_2024/images')",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token. Falls back to HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variables.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to download (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip re-downloading images that already exist on disk.",
    )
    parser.add_argument(
        "--compression",
        choices=["png", "jpg"],
        default="png",
        help="File format to save images (default: png).",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file to store additional metadata per sample.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode to load dataset iteratively (prevents OOM errors on large datasets).",
    )
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable streaming mode (not recommended for large datasets).",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save checkpoint manifest every N samples (default: 100). Set to 0 to disable checkpointing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if it exists.",
    )
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable HuggingFace dataset caching to reduce memory footprint.",
    )
    # Snapshot (full repository) mode
    parser.add_argument(
        "--snapshot-all",
        action="store_true",
        help="Download the entire dataset repository via snapshot_download and exit.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="Deepfake-Eval-2024",
        help="Local directory to place the repository snapshot (default: Deepfake-Eval-2024)",
    )
    parser.add_argument(
        "--allow-patterns",
        default=None,
        help="Comma-separated glob patterns to include in snapshot (optional)",
    )
    parser.add_argument(
        "--ignore-patterns",
        default=None,
        help="Comma-separated glob patterns to exclude from snapshot (optional)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max parallel workers for snapshot download (default: 8)",
    )
    parser.add_argument(
        "--snapshot-use-symlinks",
        action="store_true",
        help="Use symlinks for snapshot files to save disk space (default: False)",
    )
    parser.add_argument(
        "--no-resume-snapshot",
        action="store_true",
        help="Disable resume for snapshot download (default: resume enabled)",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Enable hf_transfer acceleration for snapshot (sets HF_HUB_ENABLE_HF_TRANSFER=1)",
    )
    # NEW: Restrict shards mode to avoid resolving/downloading thousands of files
    parser.add_argument(
        "--restrict-shards",
        action="store_true",
        help="Only download and load a small number of split shards instead of resolving all files.",
    )
    parser.add_argument(
        "--shard-match",
        default="train",
        help="Substring to match shard filenames for the split (default: 'train').",
    )
    parser.add_argument(
        "--shard-ext",
        choices=["parquet", "jsonl", "json", "csv"],
        default=None,
        help="Preferred shard file extension (auto-detect if omitted).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=1,
        help="Maximum number of shards to use in restrict-shards mode (default: 1).",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List remote repository files and exit (helps choosing --shard-match / --shard-ext)",
    )
    parser.add_argument(
        "--shard-file",
        default=None,
        help="Exact remote filename to download (overrides --shard-match). Use with --restrict-shards.",
    )
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_label(
    value: Any,
    encoder_cache: Dict[str, int],
    label_features: Optional[Any],
    label_map: Optional[Dict[str, int]] = None,
) -> int:
    """Encode label value to integer.

    Args:
        value: The raw label value
        encoder_cache: Cache for auto-discovered labels
        label_features: HF dataset feature info
        label_map: Explicit label -> int mapping (highest priority)

    Returns:
        Encoded label as integer
    """
    if label_map:
        key = str(value)
        if key in label_map:
            return label_map[key]
        else:
            logger.warning("Label '%s' not in provided label_map %s, falling back to encoder", key, label_map)

    if isinstance(value, (int, float)):
        return int(value)

    if label_features is not None and hasattr(label_features, "names"):
        try:
            return int(label_features.str2int(str(value)))
        except (ValueError, KeyError):
            pass

    key = str(value)
    if key not in encoder_cache:
        encoder_cache[key] = len(encoder_cache)
        logger.debug("Discovered new label '%s' -> %d", key, encoder_cache[key])
    return encoder_cache[key]


def save_manifest(manifest_rows: list[Dict[str, Any]], manifest_path: Path) -> None:
    if not manifest_rows:
        logger.warning("No rows collected; manifest not written.")
        return

    fieldnames = sorted({k for row in manifest_rows for k in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    logger.info("Manifest saved: %s (%d rows)", manifest_path, len(manifest_rows))


def load_checkpoint_manifest(checkpoint_path: Path) -> tuple[list[Dict[str, Any]], set[str]]:
    """Load existing checkpoint manifest and return rows + processed sample IDs."""
    if not checkpoint_path.exists():
        return [], set()

    rows = []
    processed_ids = set()
    try:
        with checkpoint_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                processed_ids.add(row.get("image_path", ""))
        logger.info("Loaded checkpoint from %s (%d rows)", checkpoint_path, len(rows))
        return rows, processed_ids
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s. Starting fresh.", checkpoint_path, e)
        return [], set()


def get_checkpoint_path(output_dir: Path) -> Path:
    """Get the checkpoint manifest path for resuming."""
    return output_dir / "manifest_checkpoint.csv"


# NEW: choose and download a minimal set of shards for a split
def _select_and_download_shards(
    repo_id: str,
    split_hint: str,
    shard_ext: str | None,
    max_shards: int,
    token: str | None,
) -> tuple[list[str], str]:
    """Return (local_paths, ext) for a small subset of split shards.

    This avoids resolving/downloading thousands of files to reduce memory usage.
    """
    if list_repo_files is None or hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub not available. Install with `pip install huggingface_hub`."
        )

    files = list_repo_files(repo_id, repo_type="dataset")
    # Preference order (can be overridden by shard_ext)
    ext_order = ["parquet", "jsonl", "json", "csv"]
    if shard_ext and shard_ext in ext_order:
        ext_order = [shard_ext] + [e for e in ext_order if e != shard_ext]

    chosen: list[str] = []
    used_ext: str | None = None
    # Heuristics for preferred tokens by hint
    hint = (split_hint or "").lower()
    prefer_tokens = []
    exclude_tokens = []
    if "image" in hint:
        prefer_tokens = ["image", "img", "photo", "picture"]
        exclude_tokens = ["audio"]
    elif "video" in hint:
        prefer_tokens = ["video", "frame"]
        exclude_tokens = ["audio"]
    elif "audio" in hint:
        prefer_tokens = ["audio"]

    for ext in ext_order:
        pat = re.compile(re.escape(split_hint), re.IGNORECASE) if split_hint else None
        cands = [
            f for f in files
            if f.lower().endswith("." + ext)
            and (pat.search(f) if pat else True)
        ]
        # If no direct matches, broaden search to any with this extension
        if not cands:
            cands = [f for f in files if f.lower().endswith("." + ext)]
        # Apply preferred/excluded tokens ordering
        if cands and prefer_tokens:
            pref = [f for f in cands if any(t in f.lower() for t in prefer_tokens)]
            nonpref = [f for f in cands if f not in pref]
            cands = pref + nonpref
        if cands and exclude_tokens:
            cands = [f for f in cands if not any(t in f.lower() for t in exclude_tokens)] or cands
        if cands:
            chosen = cands[: max(1, int(max_shards))]
            used_ext = ext
            break

    if not chosen or not used_ext:
        raise RuntimeError(
            f"No matching shards for split_hint='{split_hint}', ext_order={ext_order}. Total files={len(files)}."
        )

    local_paths: list[str] = []
    for fname in chosen:
        local_paths.append(
            hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fname, token=token)
        )
    return local_paths, used_ext


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    token = resolve_token(args.token)
    # Optional: list remote files and exit (introspection helper)
    if args.list_files:
        if list_repo_files is None:
            print("huggingface_hub not installed. Run: pip install huggingface_hub", flush=True)
            return 1
        files = list_repo_files(args.dataset_id, repo_type="dataset")
        print(f"Total files: {len(files)}")
        for f in files:
            print(f)
        return 0

    # Snapshot (full repository) mode
    if args.snapshot_all:
        if snapshot_download is None:
            logger.error("huggingface_hub not available. Install with `pip install huggingface_hub`.")
            return 1
        if args.enable_hf_transfer:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        try:
            if token and login is not None:
                try:
                    login(token=token)
                except Exception:
                    pass
            allow_patterns = [p.strip() for p in args.allow_patterns.split(",")] if args.allow_patterns else None
            ignore_patterns = [p.strip() for p in args.ignore_patterns.split(",")] if args.ignore_patterns else None
            local_dir = str(Path(args.snapshot_dir))
            logger.info(
                "Starting snapshot_download to %s (allow=%s ignore=%s)",
                local_dir, allow_patterns, ignore_patterns,
            )
            snapshot_download(
                repo_id=args.dataset_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=bool(args.snapshot_use_symlinks),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                max_workers=int(args.max_workers),
                resume_download=not bool(args.no_resume_snapshot),
            )
            logger.info("Snapshot completed: %s", local_dir)
            return 0
        except Exception as exc:
            logger.exception("Snapshot download failed: %s", exc)
            return 1
    output_dir = ensure_dir(Path(args.output_dir))
    images_dir = ensure_dir(output_dir / "images")
    manifest_path = output_dir / "manifest.csv"
    checkpoint_path = get_checkpoint_path(output_dir)
    metadata_path = Path(args.metadata) if args.metadata else None
    metadata_records: Dict[str, Dict[str, Any]] = {}

    # Parse label mapping if provided
    label_map: Optional[Dict[str, int]] = None
    if args.label_map:
        try:
            label_map = {}
            pairs = args.label_map.split(",")
            for pair in pairs:
                key, val = pair.strip().split("=")
                label_map[key.strip()] = int(val.strip())
            logger.info("Using custom label mapping: %s", label_map)
        except (ValueError, KeyError) as e:
            logger.error("Invalid label_map format '%s': %s", args.label_map, e)
            return 1

    # Handle resume from checkpoint
    manifest_rows: list[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    resume_idx = 0
    if args.resume:
        manifest_rows, processed_ids = load_checkpoint_manifest(checkpoint_path)
        resume_idx = len(manifest_rows)
        if manifest_rows:
            logger.info("Resuming from checkpoint with %d already processed samples", resume_idx)

    logger.info(
        "Downloading dataset '%s' (config=%s, split=%s, streaming=%s)",
        args.dataset_id,
        args.config or "<default>",
        args.split,
        args.streaming,
    )

    load_dataset_kwargs: Dict[str, Any] = {
        "path": args.dataset_id,
        "name": args.config,
        "split": args.split,
        "streaming": args.streaming,
    }
    if token:
        load_dataset_kwargs["token"] = token
    if args.disable_caching:
        load_dataset_kwargs["keep_in_memory"] = False

    ds = None  # NEW: allow alternate load path
    if args.restrict_shards:
        logger.info(
            "Restricting to %d shard(s) matching '%s' to avoid large resolution...",
            args.max_shards,
            args.shard_match,
        )
        try:
            if args.shard_file:
                if list_repo_files is None or hf_hub_download is None:
                    logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
                    return 1
                files = list_repo_files(args.dataset_id, repo_type="dataset")
                if args.shard_file not in files:
                    logger.error("--shard-file '%s' not found in repository. Use --list-files to inspect.", args.shard_file)
                    return 1
                local_paths = [
                    hf_hub_download(repo_id=args.dataset_id, repo_type="dataset", filename=args.shard_file, token=token)
                ]
                used_ext = Path(args.shard_file).suffix.lstrip(".").lower() or (args.shard_ext or "csv")
            else:
                local_paths, used_ext = _select_and_download_shards(
                    repo_id=args.dataset_id,
                    split_hint=str(args.split).split("[")[0],
                    shard_ext=args.shard_ext,
                    max_shards=args.max_shards,
                    token=token,
                )
            builder = {"parquet": "parquet", "jsonl": "json", "json": "json", "csv": "csv"}[used_ext]
            ds = load_dataset(builder, data_files={"train": local_paths}, split="train", streaming=False)
        except Exception as exc:
            logger.error("Restricted-shard loading failed: %s", exc)
            return 1

    if ds is None:
        try:
            ds = load_dataset(**load_dataset_kwargs)
        except TypeError:
            # Older datasets versions expect use_auth_token instead of token
            if token:
                load_dataset_kwargs.pop("token", None)
                load_dataset_kwargs["use_auth_token"] = token
            try:
                ds = load_dataset(**load_dataset_kwargs)
            except TypeError:
                # Fallback: remove streaming parameter for older versions
                load_dataset_kwargs.pop("streaming", None)
                ds = load_dataset(**load_dataset_kwargs)

    # Defer casting until we resolve the actual image column name below

    # Resolve columns dynamically in case CSV uses different names
    available_cols = list(getattr(ds, "column_names", []))
    logger.info("Available columns: %s", available_cols)
    def _resolve_col(preferred: str, candidates: list[str]) -> str:
        if preferred in available_cols:
            return preferred
        for c in candidates:
            if c in available_cols:
                return c
        return preferred  # fallback; may be missing

    # Try to adapt to common schemas
    image_col = _resolve_col(args.image_column, [
        "image", "image_url", "url", "path", "image_path", "filepath", "file_path"
    ])
    label_col = _resolve_col(args.label_column, [
        "label", "fake", "is_fake", "target", "y", "class"
    ])

    # Try to cast to Image only if non-streaming AND the column exists AND it's not a URL-like column
    image_col_l = image_col.lower()
    looks_like_url = any(t in image_col_l for t in ["url", "link", "media"])
    if ((not args.streaming) or args.restrict_shards) and image_col in available_cols and not looks_like_url:
        try:
            ds = ds.cast_column(image_col, Image())
        except Exception:
            # It's fine if casting fails (e.g., URLs), we'll handle formats below
            pass

    label_features = ds.features.get(label_col) if hasattr(ds, "features") else None
    label_encoder: Dict[str, int] = {}

    compression = args.compression.lower()
    if compression == "jpg":
        compression = "jpeg"

    # Prepare HTTP session for URL downloads (set UA to avoid 403s on some CDNs)
    http_session = None
    if requests is not None:
        try:
            http_session = requests.Session()
            http_session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36"
            })
        except Exception:
            http_session = None

    try:
        for idx, sample in enumerate(ds):
            # Skip samples if resuming
            if idx < resume_idx:
                continue

            if args.max_samples is not None and idx >= args.max_samples:
                break

            image = sample.get(image_col)
            if image is None:
                logger.warning("Sample %d missing image column '%s'; skipping", idx, image_col)
                continue

            # Handle PIL Image conversion in streaming mode
            if args.streaming and isinstance(image, dict):
                try:
                    from PIL import Image as PILImage
                    image = PILImage.open(image.get("path") or image.get("bytes"))
                except Exception as e:
                    logger.warning("Sample %d failed to open image: %s; skipping", idx, e)
                    continue

            label_raw = sample.get(label_col)
            label = encode_label(label_raw, label_encoder, label_features, label_map=label_map)

            label_dir = ensure_dir(images_dir / f"class_{label}")
            file_name = f"{idx:07d}.{compression}"
            image_path = label_dir / file_name

            if image_path.exists() and args.skip_existing:
                logger.debug("Skipping existing image: %s", image_path)
            else:
                # Convert and save image from various representations
                pil_image = None
                try:
                    if hasattr(image, "convert"):
                        pil_image = image.convert("RGB")  # PIL.Image
                    elif isinstance(image, dict):
                        # Could be {'path': ..., 'bytes': ...}
                        p = image.get("path")
                        b = image.get("bytes")
                        if p and os.path.exists(p):
                            from PIL import Image as PILImage  # lazy import
                            pil_image = PILImage.open(p).convert("RGB")
                        elif b is not None:
                            from PIL import Image as PILImage
                            pil_image = PILImage.open(io.BytesIO(b)).convert("RGB")
                    elif isinstance(image, str):
                        # Local path or URL
                        if image.startswith("http://") or image.startswith("https://"):
                            if http_session is None:
                                raise RuntimeError("requests not available to fetch URL images")
                            r = http_session.get(image, timeout=(5, 20), allow_redirects=True)
                            r.raise_for_status()
                            from PIL import Image as PILImage
                            pil_image = PILImage.open(io.BytesIO(r.content)).convert("RGB")
                        elif os.path.exists(image):
                            from PIL import Image as PILImage
                            pil_image = PILImage.open(image).convert("RGB")
                    # Fallback: unsupported type
                    if pil_image is None:
                        logger.warning("Sample %d: unsupported image type (%s); skipping", idx, type(image))
                        continue
                    pil_image.save(image_path, quality=95 if compression == "jpeg" else None)
                except Exception as exc:
                    logger.warning("Sample %d: failed to save image (%s); skipping", idx, exc)
                    continue

            rel_path = image_path.relative_to(output_dir)
            rel_path_str = str(rel_path).replace("\\", "/")

            # Prepend root prefix if provided
            if args.root_prefix:
                rel_path_str = args.root_prefix.rstrip("/") + "/" + rel_path_str

            # Skip if already processed (for resume functionality)
            if rel_path_str in processed_ids:
                logger.debug("Skipping already processed sample: %s", rel_path_str)
                continue

            manifest_row = {
                "image_path": rel_path_str,
                "label": int(label),
                "source_split": args.split,
            }
            manifest_rows.append(manifest_row)

            if metadata_path:
                metadata_records[rel_path_str] = {
                    "label_raw": str(label_raw),
                    "label_encoded": int(label),
                }

            # Periodic checkpoint save
            if args.checkpoint_freq > 0 and (idx + 1) % args.checkpoint_freq == 0:
                save_manifest(manifest_rows, checkpoint_path)
                logger.info("Checkpoint saved: %d samples processed", idx + 1)

            # Log progress (without total count in streaming mode)
            if args.streaming:
                if (idx + 1) % 100 == 0:
                    logger.info("Processed %d samples...", idx + 1)
            else:
                total = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
                if (idx + 1) % 100 == 0 or (idx + 1) == total:
                    logger.info("Processed %d/%d samples", idx + 1, total)

    except Exception as e:
        if "bad_alloc" in str(e).lower() or "out of memory" in str(e).lower():
            logger.error(
                "Memory exhaustion error: %s\n"
                "This usually happens with large datasets when not using streaming mode.\n"
                "Try using --streaming (enabled by default) or --max-samples to limit the download.",
                e,
            )
            return 1
        else:
            logger.exception("Unexpected error during dataset processing: %s", e)
            return 1

    # Save final manifest
    save_manifest(manifest_rows, manifest_path)

    # Clean up checkpoint file after successful completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            logger.debug("Removed checkpoint file: %s", checkpoint_path)
        except Exception as e:
            logger.warning("Could not remove checkpoint file %s: %s", checkpoint_path, e)

    if metadata_path and metadata_records:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_records, f, indent=2)
        logger.info("Sample metadata saved: %s", metadata_path)

    logger.info("Done. Images stored under %s (%d samples processed)", images_dir, len(manifest_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
