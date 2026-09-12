# Pinned versions (snapshot taken 2026-09-12)

Everything below is what the code in this repository was validated against. Change any of them
only with a note in this file and a re-run of the smoke test.

## Source code

| Component | Location | Revision | Notes |
|---|---|---|---|
| VCD (official) | https://github.com/DAMO-NLP-SG/VCD | `d6568ff81b8fd306a49e630df44f2db5c2300191` (2024-10-07) | cloned to `third_party/VCD` (git-ignored). Uses `vcd_utils/vcd_sample.py` (monkeypatches `GenerationMixin.sample`) and `vcd_utils/vcd_add_noise.py`. Vendored LLaVA lives in `experiments/llava` and is imported via `sys.path`, not installed. |
| VLMBias generators (official) | https://github.com/anvo25/vlms-are-biased | `c8aaa69c71c66ed3883ce69c1e858f3862f1e5c9` (2026-01-26) | cloned to `third_party/vlms-are-biased`. Note: the released dataset prompts differ from HEAD (e.g. released Ebbinghaus Q1 says "two red circles"; HEAD source says "two inner circles"). Treat the HF dataset revision below, not the generator, as the source of truth for prompts. |

## Model

| Item | Value |
|---|---|
| Checkpoint | `liuhaotian/llava-v1.5-7b` |
| HF revision | `4481d270cc22fd5c4d1bb5df129622006ccd9234` |
| Weights | `pytorch_model-00001-of-00002.bin` (9.98 GB) + `pytorch_model-00002-of-00002.bin` (3.54 GB) + `mm_projector.bin`; 13.56 GB total |
| Vision tower (loaded separately by LLaVA) | `openai/clip-vit-large-patch14-336` @ `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| Model config facts | `image_aspect_ratio: pad`, `mm_use_im_start_end: false`, `mm_projector_type: mlp2x_gelu`, `torch_dtype: float16` |
| Conversation template | `llava_v1` (as in the VCD LLaVA eval script) |
| Precision / device | FP16, batch size 1, single RTX A5000 (24 GB) |

## Dataset

| Item | Value |
|---|---|
| Dataset | `anvo25/vlms-are-biased` (Hugging Face) |
| HF revision | `3761f9fd7163577534a7816c8bc1004035f2e2a0` |
| Splits used | `main` (2,784 rows = 464 cases x 3 resolutions x Q1/Q2). See `data/inventory/summary.md`. |
| Pair manifest | `data/manifests/optical_pairs_v0.1.jsonl` (60 pairs, 22 templates; Q1, 768 px) |

## Python environment (`conda env: vcd`)

| Package | Version | Why |
|---|---|---|
| python | 3.10 | transformers 4.31.0 does not build on the machine's default 3.13 |
| torch | 2.0.1+cu118 | VCD requirement; the cu118 wheel is the only CUDA 11.x/12.x build published for 2.0.1 |
| torchvision | 0.15.2+cu118 | VCD requirement |
| transformers | 4.31.0 | VCD monkeypatch targets this generation loop |
| tokenizers | 0.13.3 | `<0.14` per VCD |
| accelerate | 0.21.0 | VCD requirement |
| sentencepiece | 0.1.99 | VCD requirement |
| numpy | 1.26.4 | numpy 2 breaks torch 2.0.1 |
| huggingface_hub | 0.25.2 | transformers 4.31 requires `<1.0` |
| datasets | 3.0.1 | reading the parquet snapshot |

Full lock: `environment.lock.txt` (`pip freeze`).

## System

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A5000, 24 GB (4 on the node; runs pinned to one via `CUDA_VISIBLE_DEVICES`) |
| Driver | 580.126.09 |
| cuDNN (bundled with torch wheel) | 8700 |
| OS | Linux 6.8.0-106-generic |

## VCD hyperparameters recorded from the official repo

| Source | cd_alpha | cd_beta | noise_step | temperature | top_p | top_k |
|---|---:|---:|---:|---:|---:|---:|
| `vcd_utils/vcd_sample.py` code defaults | 0.5 | 0.1 | - | - | - | - |
| `experiments/eval/object_hallucination_vqa_llava.py` argparse defaults | 1 | 0.1 | 500 | 1.0 | 1 | None |
| `experiments/cd_scripts/llava1.5_pope.bash` | 1 | 0.2 | 500 | 1.0 | 1 | None |

The EnAR (CVPR 2026) setting used to produce the 16.92 -> 19.18 figure has **not** yet been confirmed;
`configs/smoke_optical.yaml` follows the run script (1 / 0.2 / 500) until it is.
