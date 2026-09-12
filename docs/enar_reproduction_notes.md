# What the EnAR paper does and does not specify about its VCD baseline

Source: Liang et al., "Envision, Attend, Then Respond" (CVPR 2026), main paper + supplemental PDF from
openaccess.thecvf.com, and the code release https://github.com/Lyxxx1211/CVPR2026-EnAR (commit `815f44f`,
2026-06-23). Checked 2026-09-12.

## Hyperparameters: not published

- Main text §4.1: *"Our contrastive decoding implementation and hyperparameter setting follow the
  configuration of VCD. More details can be found in Appendix B.2."*
- Appendix B.2 contains only the VCD formula, written in probability space:
  `p_vcd = (1+α) p(y|v,x) − α p(y|v',x)` (Eq. 16). No α, β (APC), noise step, temperature, top-p/k,
  max-new-tokens, or seeds are given anywhere in the paper or supplement.
- The code release README states it *"does not keep ... LLaVA/Qwen comparison code"*; the repo contains
  only the InternVL + Stable Diffusion EnAR pipeline. Its own runner uses `do_sample=False` for EnAR.
- "The configuration of VCD" is itself ambiguous: the official VCD repo has three settings
  (code defaults α=0.5/β=0.1; eval-script argparse α=1/β=0.1/noise 500; run script α=1/β=0.2/noise 500).
  See VERSIONS.md.

**Decision:** keep `configs/smoke_optical.yaml` at the VCD run-script setting (α=1, β=0.2, noise_step=500,
temperature=1.0, top_p=1, top_k=None) and treat α/β as a declared sensitivity axis rather than a known
constant. Consider emailing the authors (Fudan; addresses on the paper) for the exact LLaVA/VCD command.

## Evaluation denominator: 928 rows, not 2,784

Reverse-engineering Table 1 (LLaVA-v1.5-7B row): every per-domain accuracy is an exact integer ratio over
`2 x unique cases`, and the totals reproduce the headline numbers exactly.

| domain | unique cases | denominator | Regular | VCD |
|---|---:|---:|---|---|
| Animals | 91 | 182 | 0/182 = 0.00 | 2/182 = 1.09 |
| Chess Pieces | 48 | 96 | 0/96 | 0/96 |
| Flags | 40 | 80 | 7/80 = 8.75 | 6/80 = 7.50 |
| Game Boards | 28 | 56 | 6/56 = 10.71 | 3/56 = 5.36 |
| Logos | 69 | 138 | 12/138 = 8.70 | 19/138 = 13.77 |
| Optical Illusion | 132 | 264 | 132/264 = 50.00 | 127/264 = 48.11 |
| Patterned Grid | 56 | 112 | 0/112 = 0.00 | 21/112 = 18.75 |
| **Overall** | 464 | **928** | **157/928 = 16.92** | **178/928 = 19.18** |

928 = 2784 / 3 = one resolution x both prompt forms (Q1 and Q2). The `main` split has exactly 928 rows at
each of 384, 768, and 1152 px, so the resolution EnAR used cannot be identified from the table.

Implications for SPEC §5 (proposed amendments, not yet applied):
1. The "aggregate reproduction" target is a 928-row subset (one resolution, Q1+Q2), not all 2,784 rows.
   Run all three resolutions (3 x 928 rows, ~1 h on one A5000 per condition) and report which, if any,
   matches; the full 2,784-row run is then a by-product.
2. Q1 and Q2 of the same image are *not* independent in their denominator either; the deduplicated
   analysis view (one prompt, one resolution per case: 464 rows) is still required for inference.
3. Note that in EnAR's table VCD *lowers* LLaVA's Optical Illusion accuracy (50.00 -> 48.11) and Flags
   (8.75 -> 7.50); the overall gain comes almost entirely from Patterned Grid (0 -> 21/112) and Logos.
   The optical-illusion pairs in this repo are therefore a domain where the published VCD effect is
   negative, which matters for how the paired result is framed.
4. Regular = 0/182 on Animals and 0/112 on Patterned Grid with sampling suggests a strict parser
   (e.g. exact `{number}` match) or a small `max_new_tokens`; parser leniency is a discrepancy candidate.
