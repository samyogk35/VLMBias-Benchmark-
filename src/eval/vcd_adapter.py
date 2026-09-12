"""Thin, instrumented adapter around the official VCD code and its vendored LLaVA-1.5.

Everything VCD-specific is isolated here so that a silent failure of the monkeypatch is
detectable (SPEC §11/§12):

  * ``patch_status()`` checks that ``GenerationMixin.sample`` really is VCD's function.
  * ``generate_once`` wraps ``prepare_inputs_for_generation_cd`` with a call counter, so a VCD
    run that never touched the distorted image raises instead of quietly returning the
    regular-sampling answer.
  * ``first_step_diagnostics`` computes clean vs. noised next-token logits and the VCD
    combination outside the generation loop, so a nonzero regular-vs-contrastive score
    difference can be asserted and stored.

Decoding conditions implemented:
  greedy   -- do_sample=False (transformers greedy_search; VCD patch not involved)
  regular  -- do_sample=True, no images_cd (VCD's patched sample() takes its non-CD branch)
  vcd      -- do_sample=True with images_cd, cd_alpha, cd_beta (full VCD incl. APC)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VCD_ROOT = os.path.join(ROOT, "third_party", "VCD")
for p in (VCD_ROOT, os.path.join(VCD_ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import transformers  # noqa: E402
from transformers import set_seed  # noqa: E402
from vcd_utils import vcd_sample as _vcd_sample  # noqa: E402
from vcd_utils.vcd_add_noise import add_diffusion_noise  # noqa: E402
from vcd_utils.vcd_sample import evolve_vcd_sampling  # noqa: E402
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN  # noqa: E402
from llava.conversation import conv_templates, SeparatorStyle  # noqa: E402
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria  # noqa: E402

MODEL_REPO = "liuhaotian/llava-v1.5-7b"
MODEL_REVISION = "4481d270cc22fd5c4d1bb5df129622006ccd9234"

CONDITIONS = ("greedy", "regular", "vcd")


def resolve_snapshot(repo: str = MODEL_REPO, revision: str = MODEL_REVISION) -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    d = os.path.join(hf_home, "hub", f"models--{repo.replace('/', '--')}", "snapshots", revision)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"model snapshot not found at {d}")
    return d


def install_patch() -> None:
    """Install VCD's sampling monkeypatch (idempotent).

    Also repairs an upstream bug: vcd_sample.py references ``SampleDecoderOnlyOutput`` /
    ``SampleEncoderDecoderOutput`` without importing them (only hit with
    ``return_dict_in_generate=True``, which the official scripts never use). We inject the names
    rather than editing third_party/.
    """
    from transformers.generation.utils import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
    _vcd_sample.SampleDecoderOnlyOutput = SampleDecoderOnlyOutput
    _vcd_sample.SampleEncoderDecoderOutput = SampleEncoderDecoderOutput
    evolve_vcd_sampling()


def patch_status() -> dict:
    gm = transformers.generation.utils.GenerationMixin
    return {
        "sample_is_vcd": gm.sample is _vcd_sample.sample,
        "sample_qualname": f"{gm.sample.__module__}.{gm.sample.__qualname__}",
        "transformers_version": transformers.__version__,
    }


class LlavaRunner:
    def __init__(self, snapshot_dir: Optional[str] = None, conv_mode: str = "llava_v1", device: str = "cuda"):
        disable_torch_init()
        self.snapshot_dir = snapshot_dir or resolve_snapshot()
        self.conv_mode = conv_mode
        self.device = device
        model_name = get_model_name_from_path(self.snapshot_dir)  # -> derives "llava-v1.5-7b" style name
        if "llava" not in model_name.lower():
            model_name = "llava-v1.5-7b"
        # device_map pins the whole model to one GPU (batch size 1, FP16)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.snapshot_dir, None, model_name, device_map={"": 0}, device=device)
        self.model.eval()
        self.image_aspect_ratio = getattr(self.model.config, "image_aspect_ratio", None)

    # ---- inputs -------------------------------------------------------------------------
    def build_prompt(self, question: str) -> str:
        qs = question
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt(), conv

    def encode(self, question: str, pil_image):
        prompt, conv = self.build_prompt(question)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        # LLaVA-1.5 was trained with `image_aspect_ratio: pad` (expand2square). The VCD eval script
        # calls image_processor.preprocess directly (centre-crop); we follow the model config instead
        # and record the choice in the run config.
        image_tensor = process_images([pil_image.convert("RGB")], self.image_processor, self.model.config)[0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return prompt, input_ids, image_tensor, stop_str

    # ---- generation ---------------------------------------------------------------------
    @torch.inference_mode()
    def generate_once(self, question: str, pil_image, condition: str, seed: int, max_new_tokens: int = 16,
                      temperature: float = 1.0, top_p: float = 1.0, top_k: Optional[int] = None,
                      cd_alpha: float = 1.0, cd_beta: float = 0.2, noise_step: int = 500,
                      save_first_step_topk: int = 10) -> dict:
        assert condition in CONDITIONS, condition
        if condition == "vcd" and not patch_status()["sample_is_vcd"]:
            raise RuntimeError("VCD monkeypatch is not installed; call install_patch() first")

        prompt, input_ids, image_tensor, stop_str = self.encode(question, pil_image)
        images = image_tensor.unsqueeze(0).half().to(self.device)

        set_seed(seed)
        images_cd = None
        if condition == "vcd":
            images_cd = add_diffusion_noise(image_tensor, noise_step).unsqueeze(0).half().to(self.device)

        # instrument the distorted-image branch
        n_cd_calls = 0
        orig_cd = self.model.prepare_inputs_for_generation_cd

        def counting_cd(*a, **k):
            nonlocal n_cd_calls
            n_cd_calls += 1
            return orig_cd(*a, **k)

        self.model.prepare_inputs_for_generation_cd = counting_cd
        stopping = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
        gen_kwargs = dict(
            images=images,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            return_dict_in_generate=True,
            output_scores=True,
        )
        if condition == "greedy":
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        if condition == "vcd":
            gen_kwargs.update(images_cd=images_cd, cd_alpha=cd_alpha, cd_beta=cd_beta)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = self.model.generate(input_ids, **gen_kwargs)
        finally:
            self.model.prepare_inputs_for_generation_cd = orig_cd
        torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - t0) * 1000.0

        seq = out.sequences
        n_in = input_ids.shape[1]
        new_ids = seq[:, n_in:]
        text = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        n_new = int(new_ids.shape[1])

        if condition == "vcd" and n_cd_calls == 0:
            raise RuntimeError("VCD condition ran but the distorted-image branch was never executed")
        if condition != "vcd" and n_cd_calls != 0:
            raise RuntimeError(f"non-VCD condition {condition} executed the distorted-image branch {n_cd_calls}x")

        # first-step scores (post-processing: for vcd these are the contrastive logits after APC)
        first_topk = None
        if out.scores:
            s0 = out.scores[0][0].float()
            finite = torch.isfinite(s0)
            vals, idx = torch.topk(torch.where(finite, s0, torch.full_like(s0, -1e30)), k=save_first_step_topk)
            first_topk = [(self.tokenizer.convert_ids_to_tokens(int(i)), float(v)) for v, i in zip(vals, idx)]
            n_alive = int(finite.sum())
        else:
            n_alive = None

        return {
            "prompt": prompt,
            "raw_output": text,
            "n_new_tokens": n_new,
            "n_cd_forward_calls": n_cd_calls,
            "hit_max_new_tokens": n_new >= max_new_tokens,
            "first_step_topk": first_topk,
            "first_step_n_unmasked": n_alive,
            "duration_ms": duration_ms,
        }

    # ---- diagnostics --------------------------------------------------------------------
    @torch.inference_mode()
    def first_step_diagnostics(self, question: str, pil_image, seed: int, cd_alpha: float, cd_beta: float,
                               noise_step: int, topk: int = 5) -> dict:
        """Clean vs. noised next-token logits at the first decoding step and the VCD combination.

        Independent of the generation loop; used to prove the contrastive term is nonzero.
        """
        prompt, input_ids, image_tensor, _ = self.encode(question, pil_image)
        images = image_tensor.unsqueeze(0).half().to(self.device)
        set_seed(seed)
        images_cd = add_diffusion_noise(image_tensor, noise_step).unsqueeze(0).half().to(self.device)

        clean = self.model(input_ids=input_ids, images=images, use_cache=False).logits[0, -1].float()
        noised = self.model(input_ids=input_ids, images=images_cd, use_cache=False).logits[0, -1].float()
        cutoff = torch.log(torch.tensor(cd_beta)) + clean.max()
        vcd_logits = (1 + cd_alpha) * clean - cd_alpha * noised
        vcd_logits = vcd_logits.masked_fill(clean < cutoff, -float("inf"))

        def top(t):
            v, i = torch.topk(torch.where(torch.isfinite(t), t, torch.full_like(t, -1e30)), k=topk)
            return [(self.tokenizer.convert_ids_to_tokens(int(a)), round(float(b), 3)) for b, a in zip(v, i)]

        p_clean = torch.softmax(clean, -1)
        p_noised = torch.softmax(noised, -1)
        p_vcd = torch.softmax(vcd_logits, -1)
        return {
            "clean_top": top(clean),
            "noised_top": top(noised),
            "vcd_top": top(vcd_logits),
            "logit_l1_clean_vs_noised": float((clean - noised).abs().sum()),
            "prob_l1_clean_vs_vcd": float((p_clean - p_vcd).abs().sum()),
            "prob_l1_clean_vs_noised": float((p_clean - p_noised).abs().sum()),
            "apc_tokens_kept": int(torch.isfinite(vcd_logits).sum()),
            "image_pixel_l2_clean_vs_noised": float((images.float() - images_cd.float()).norm()),
        }
