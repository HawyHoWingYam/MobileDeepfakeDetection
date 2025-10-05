"""Pytest smoke tests for Stage 02 datasets."""

import csv
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

STAGE2_ROOT = SRC_ROOT / 'stage_02'
if str(STAGE2_ROOT) not in sys.path:
    sys.path.append(str(STAGE2_ROOT))

from src.stage_02.train_stage2_spatial import SpatialExpertDataset, AugmentationConfig as SpatialAugConfig
from src.stage_02.train_stage2_genconvit import GenConViTDataset, AugmentationConfig as GenConAugConfig


@pytest.fixture()
def manifest_root(tmp_path):
    root = tmp_path
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    img_path = image_dir / "sample.png"
    Image.new('RGB', (64, 64), color=(128, 128, 128)).save(img_path)

    manifest_path = root / "train.csv"
    with manifest_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image_path', 'label', 'split', 'valid']
        )
        writer.writeheader()
        writer.writerow({
            'image_path': str(Path('images/sample.png')),
            'label': 0,
            'split': 'train',
            'valid': True,
        })

    return root, manifest_path


def test_spatial_manifest_dataset(manifest_root):
    root, manifest = manifest_root
    dataset = SpatialExpertDataset(
        split='train',
        resolution=64,
        augmentation_config=SpatialAugConfig(spatial_expert_mode=True),
        manifest_path=str(manifest),
        dataset_root=root,
        fallback_path=None,
    )
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape[0] == 3
    assert label.item() == pytest.approx(0.0)


def test_genconvit_manifest_dataset(manifest_root):
    root, manifest = manifest_root
    dataset = GenConViTDataset(
        split='train',
        resolution=64,
        augmentation_config=GenConAugConfig(generative_expert_mode=True),
        return_reconstruction_target=True,
        manifest_path=str(manifest),
        dataset_root=root,
        fallback_path=None,
    )
    processed, target, label = dataset[0]
    assert isinstance(processed, torch.Tensor)
    assert processed.shape == target.shape
    assert label.item() == pytest.approx(0.0)
