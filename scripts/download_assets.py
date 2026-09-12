"""Download the pinned model, vision tower, and dataset snapshots into the HF cache."""
from huggingface_hub import snapshot_download

PINS = [
    ("anvo25/vlms-are-biased", "dataset", "3761f9fd7163577534a7816c8bc1004035f2e2a0", None),
    ("openai/clip-vit-large-patch14-336", "model", "ce19dc912ca5cd21c8a653c79e251e808ccabcd1", ["*.json", "*.txt", "pytorch_model.bin"]),
    ("liuhaotian/llava-v1.5-7b", "model", "4481d270cc22fd5c4d1bb5df129622006ccd9234", None),
]
for repo, rtype, rev, patterns in PINS:
    p = snapshot_download(repo, repo_type=rtype, revision=rev, allow_patterns=patterns)
    print(repo, "->", p)
