# Inventory of the VLMBias main split: how many rows per topic / prompt / resolution,
# and how many actual distinct cases there are once you collapse the duplicates.
# Writes data/inventory/main_rows.csv and data/inventory/summary.md

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.common import load_split_metadata, DATASET_REVISION

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "inventory")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_split_metadata("main")
    df.drop(columns=["metadata"]).to_csv(os.path.join(OUT_DIR, "main_rows.csv"), index=False)

    df["image_file"] = df.image_path.str.split("/").str[-1]
    # two filename styles: "..._notitle_px768.png" and "..._768.png" (animals/flags/logos)
    df["case_key"] = df.image_file.str.replace(r"(_px|_)(384|768|1152)\.png$", "", regex=True)

    lines = ["# VLMBias inventory (main split, revision %s)" % DATASET_REVISION, "",
             "Total rows: %d" % len(df), "",
             "## rows by topic / question / resolution", ""]
    piv = df.pivot_table(index=["topic", "type_of_question"], columns="pixel", values="ID", aggfunc="count", fill_value=0)
    lines += [piv.to_markdown(), ""]

    lines += ["## distinct images and cases per topic", "",
              "| topic | rows | image files | cases (resolutions collapsed) | rows per case |", "|---|---:|---:|---:|---:|"]
    for topic, g in df.groupby("topic"):
        lines.append("| %s | %d | %d | %d | %.1f |" % (topic, len(g), g.image_file.nunique(), g.case_key.nunique(), len(g) / g.case_key.nunique()))
    n_case = df.case_key.nunique()
    lines.append("| all | %d | %d | %d | %.1f |" % (len(df), df.image_file.nunique(), n_case, len(df) / n_case))
    lines += ["", "Every case shows up 6 times: 3 resolutions x 2 prompt wordings (Q1/Q2).",
              "So the 2784 rows are really 464 cases, and any statistics need to pick one resolution and one prompt.", ""]

    lines += ["## optical illusions: ground truth vs expected bias", ""]
    opt = df[df.topic == "Optical Illusion"]
    lines += [opt.groupby(["sub_topic", "ground_truth", "expected_bias"]).size().unstack([1, 2]).fillna(0).astype(int).to_markdown(), ""]

    lines += ["## other splits", ""]
    for split in ["identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]:
        d = load_split_metadata(split)
        lines.append("- %s: %d rows, topics: %s" % (split, len(d), ", ".join(sorted(d.topic.unique()))))

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
