"""Build the dataset inventory for VLMBias (SPEC §3 item 2).

Writes:
  data/inventory/main_rows.csv        one row per `main` record (no image bytes)
  data/inventory/summary.md           topic x question x resolution counts, unique images, duplication factor
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.common import load_split_metadata, DATASET_REVISION  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "inventory")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    main_df = load_split_metadata("main")
    main_df.drop(columns=["metadata"]).to_csv(os.path.join(OUT_DIR, "main_rows.csv"), index=False)

    lines = [f"# VLMBias inventory (`main` split, revision `{DATASET_REVISION}`)", "",
             f"Total rows: **{len(main_df)}**", ""]

    # unique images = unique image_path (each image appears under Q1 and Q2)
    main_df["image_file"] = main_df.image_path.str.split("/").str[-1]
    # strip the px suffix to find the underlying rendered case shared across resolutions
    # two filename conventions: "..._notitle_px768.png" (generators) and "..._768.png" (animals/flags/logos)
    main_df["case_key"] = main_df.image_file.str.replace(r"(_px|_)(384|768|1152)\.png$", "", regex=True)

    lines += ["## Rows by topic, question form, and resolution", ""]
    piv = main_df.pivot_table(index=["topic", "type_of_question"], columns="pixel", values="ID", aggfunc="count", fill_value=0)
    lines += [piv.to_markdown(), ""]

    lines += ["## Unique images and underlying cases per topic", "",
              "| topic | rows | unique image files | unique cases (px-collapsed) | rows per case |", "|---|---:|---:|---:|---:|"]
    for topic, g in main_df.groupby("topic"):
        n_img = g.image_file.nunique(); n_case = g.case_key.nunique()
        lines.append(f"| {topic} | {len(g)} | {n_img} | {n_case} | {len(g) / n_case:.1f} |")
    tot_case = main_df.case_key.nunique()
    lines.append(f"| **all** | {len(main_df)} | {main_df.image_file.nunique()} | {tot_case} | {len(main_df) / tot_case:.1f} |")
    lines += ["", "Each underlying case appears at 3 resolutions (384/768/1152) x 2 prompt forms (Q1/Q2) = 6 rows.",
              "Inferential analyses must use one prompt and one resolution per case (SPEC §5).", ""]

    lines += ["## Optical illusion sub-topics (ground truth x expected bias)", ""]
    opt = main_df[main_df.topic == "Optical Illusion"]
    lines += [opt.groupby(["sub_topic", "ground_truth", "expected_bias"]).size().unstack([1, 2]).fillna(0).astype(int).to_markdown(), ""]

    lines += ["## Other splits", ""]
    for split in ["identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]:
        try:
            d = load_split_metadata(split)
            lines.append(f"- `{split}`: {len(d)} rows, topics: {', '.join(sorted(d.topic.unique()))}")
        except FileNotFoundError as e:
            lines.append(f"- `{split}`: not available ({e})")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
