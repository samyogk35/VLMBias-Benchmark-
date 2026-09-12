# Notes on reproducing EnAR's VCD numbers

I went through the EnAR paper, its supplement and its GitHub repo (Sept 12) looking for the exact
VCD setup behind the 16.92 -> 19.18 LLaVA result.

## The hyperparameters aren't published

Section 4.1 says "Our contrastive decoding implementation and hyperparameter setting follow the
configuration of VCD. More details can be found in Appendix B.2." Appendix B.2 only has the formula,
p_vcd = (1+a) p(y|v,x) - a p(y|v',x), nothing numeric: no alpha, no beta for the plausibility
constraint, no noise step, no temperature, no seeds, no max tokens. The code release README says it
"does not keep ... LLaVA/Qwen comparison code", and indeed it's just the InternVL + Stable Diffusion
part of EnAR. Their own runner uses do_sample=False.

"The configuration of VCD" is not one thing either, the VCD repo has three different settings
(see VERSIONS.md). I'm sticking with the run-script values (alpha 1, beta 0.2, noise 500) and will
treat alpha/beta as something to vary rather than a known constant. Could email the authors.

## They evaluated 928 rows, not 2784

Working backwards from Table 1 (LLaVA-v1.5-7B row): every per-domain percentage is an exact integer
count over 2 x (number of unique cases), and the totals come out exactly.

| domain | cases | denom | Regular | VCD |
|---|---:|---:|---|---|
| Animals | 91 | 182 | 0/182 = 0.00 | 2/182 = 1.09 |
| Chess Pieces | 48 | 96 | 0/96 | 0/96 |
| Flags | 40 | 80 | 7/80 = 8.75 | 6/80 = 7.50 |
| Game Boards | 28 | 56 | 6/56 = 10.71 | 3/56 = 5.36 |
| Logos | 69 | 138 | 12/138 = 8.70 | 19/138 = 13.77 |
| Optical Illusion | 132 | 264 | 132/264 = 50.00 | 127/264 = 48.11 |
| Patterned Grid | 56 | 112 | 0/112 = 0.00 | 21/112 = 18.75 |
| total | 464 | 928 | 157/928 = 16.92 | 178/928 = 19.18 |

928 = 2784/3, i.e. one resolution with both Q1 and Q2. The main split has exactly 928 rows at each of
384, 768 and 1152 px, so I can't tell which resolution they used from the table alone.

What this changes for me:
- The reproduction target is a 928-row subset, not the whole split. Cheapest thing is to run all three
  resolutions (3 x 928 per condition, about an hour each on one A5000) and see which one lands near
  16.92 / 19.18. The full 2784 run comes for free.
- Q1 and Q2 on the same image are still not independent, so the real statistics need one prompt and
  one resolution per case (464 rows).
- In their own table VCD makes LLaVA *worse* on optical illusions (50.00 -> 48.11) and on flags. The
  overall gain is basically Patterned Grid (0 -> 21/112) plus Logos. My optical pairs are a domain where
  the published effect is negative, worth remembering when I interpret the paired result.
- Regular scoring exactly 0 on Animals and Patterned Grid with sampling suggests a strict parser or a
  tiny max_new_tokens. If my numbers don't match, parser leniency is the first thing to check.
