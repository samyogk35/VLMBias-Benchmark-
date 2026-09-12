"""Shared helpers for locating the pinned VLMBias snapshot."""
from __future__ import annotations

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

DATASET_REPO = "anvo25/vlms-are-biased"
DATASET_REVISION = "3761f9fd7163577534a7816c8bc1004035f2e2a0"

META_COLS = ["ID", "image_path", "topic", "sub_topic", "prompt", "ground_truth", "expected_bias",
             "with_title", "type_of_question", "pixel", "metadata"]


def snapshot_dir() -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    d = os.path.join(hf_home, "hub", f"datasets--{DATASET_REPO.replace('/', '--')}", "snapshots", DATASET_REVISION)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"dataset snapshot not found: {d}. Run scripts/download_assets.py first.")
    return d


def split_files(split: str):
    files = sorted(glob.glob(os.path.join(snapshot_dir(), "data", f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files for split {split!r}")
    return files


def load_split_metadata(split: str) -> pd.DataFrame:
    """Load a split without decoding image bytes."""
    frames = []
    for f in split_files(split):
        names = pq.read_schema(f).names
        cols = [c for c in META_COLS if c in names]
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["split"] = split
    return df


def iter_images(split: str, wanted_ids: set):
    """Yield (ID, PIL.Image) for rows whose ID is in wanted_ids, flattened onto white RGB."""
    import io
    from PIL import Image
    for f in split_files(split):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64, columns=["ID", "image"]):
            for rid, img in zip(batch.column("ID").to_pylist(), batch.column("image").to_pylist()):
                if rid not in wanted_ids:
                    continue
                im = Image.open(io.BytesIO(img["bytes"]))
                if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[-1])
                    im = bg
                else:
                    im = im.convert("RGB")
                yield rid, im
