# Versions

Everything the code was run against, as of 2026-09-12.

Code
- VCD: https://github.com/DAMO-NLP-SG/VCD at d6568ff81b8fd306a49e630df44f2db5c2300191 (2024-10-07).
  Cloned into third_party/VCD. Their LLaVA copy is in experiments/llava and is imported via sys.path,
  same as their own scripts do.
- VLMBias generators: https://github.com/anvo25/vlms-are-biased at c8aaa69c71c66ed3883ce69c1e858f3862f1e5c9
  (2026-01-26). Note the released dataset's prompts don't match this code exactly, the dataset is the
  source of truth.

Model
- liuhaotian/llava-v1.5-7b, HF revision 4481d270cc22fd5c4d1bb5df129622006ccd9234 (13.56 GB)
- vision tower openai/clip-vit-large-patch14-336, revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1
  (LLaVA loads it separately)
- config: image_aspect_ratio=pad, mm_use_im_start_end=false, mlp2x_gelu projector, fp16
- conversation template llava_v1, fp16, batch size 1, one RTX A5000

Dataset
- anvo25/vlms-are-biased on Hugging Face, revision 3761f9fd7163577534a7816c8bc1004035f2e2a0
- main split: 2784 rows = 464 cases x {384, 768, 1152} px x {Q1, Q2}. See data/inventory/summary.md
- pair manifest: data/manifests/optical_pairs_v0.1.jsonl, 60 pairs / 22 templates, Q1 at 768 px

Python (conda env "vcd", full pip freeze in environment.lock.txt)
- python 3.10 (3.13 can't build transformers 4.31)
- torch 2.0.1+cu118, torchvision 0.15.2+cu118 (VCD's pins; only a cu118 wheel exists for 2.0.1)
- transformers 4.31.0, tokenizers 0.13.3, accelerate 0.21.0, sentencepiece 0.1.99
- numpy 1.26.4 (2.x breaks torch 2.0.1), huggingface_hub 0.25.2, datasets 3.0.1

Machine
- 4x RTX A5000 24 GB (shared node, I use one), driver 580.126.09, cuDNN 8700, Linux 6.8.0

VCD hyperparameters found in the VCD repo (they don't agree with each other)
- vcd_sample.py defaults: alpha 0.5, beta 0.1
- eval/object_hallucination_vqa_llava.py argparse: alpha 1, beta 0.1, noise_step 500, temp 1.0, top_p 1, top_k None
- cd_scripts/llava1.5_pope.bash: alpha 1, beta 0.2, noise_step 500
I use the bash script's values. EnAR (CVPR 2026) doesn't say which it used, see docs/enar_reproduction_notes.md.
EnAR paper: https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Envision_Attend_Then_Respond_Counterfactual_Hallucination_Mitigation_in_Large_Vision-Language_CVPR_2026_paper.html
EnAR code: https://github.com/Lyxxx1211/CVPR2026-EnAR at 815f44fc8577e6f67a4f815a357b481179d5fa70 (no LLaVA/VCD code in it)
