# download the model, the CLIP vision tower and the dataset at the pinned revisions
from huggingface_hub import snapshot_download

snapshot_download("anvo25/vlms-are-biased", repo_type="dataset", revision="3761f9fd7163577534a7816c8bc1004035f2e2a0")
snapshot_download("openai/clip-vit-large-patch14-336", revision="ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
                  allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
snapshot_download("liuhaotian/llava-v1.5-7b", revision="4481d270cc22fd5c4d1bb5df129622006ccd9234")
print("done")
