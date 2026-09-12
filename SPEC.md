# Project Specification: Paired Audit of Visual Contrastive Decoding

**Version:** 0.1  
**Date:** 2026-09-12  
**Status:** Ready for implementation  
**Model:** `liuhaotian/llava-v1.5-7b`  
**Benchmark:** VLMBias  

## 1. Decision and project outcome

This is a feasible and worthwhile semester project if the paired audit, rather than a 200–300-pair target, is treated as the contribution. The project will test whether Visual Contrastive Decoding (VCD) improves a model's ability to distinguish a canonical image from its minimally modified counterfactual, or whether it merely changes the model's answer distribution in a way that happens to score better on counterfactual examples.

The project succeeds even if the result is negative. A careful reproduction plus evidence that VCD's gain is genuine, spurious, mixed, or inconclusive is a valid outcome.

## 2. Research question and hypotheses

### Primary research question

When VCD improves LLaVA-1.5-7B accuracy on VLMBias counterfactual images, does it also improve correct discrimination of matched canonical–counterfactual pairs?

### HypothesesProgress Report 1
Step 1: Work on Your Project
Continue working on your project according to the plan and timeline in your approved proposal.

Step 2: Prepare a Progress Report
Write a progress report describing what you have completed so far. Please provide specific details and evidence of your work, rather than simply stating that you worked on the project.

Your report should include:

Project Overview – Briefly describe your project and its main goal.

Work Completed – Explain in detail what you have accomplished so far.

Evidence of Progress – Include a GitHub link, screenshots, figures, results, code examples, designs, or other evidence when appropriate.

Challenges – Describe any problems you encountered and how you addressed them.

Next Steps – Explain what you plan to complete before the next progress report.

- **H1 — grounding:** VCD increases exact pair success: the model answers both the canonical and counterfactual member correctly.
- **H2 — prior avoidance:** VCD increases counterfactual accuracy but produces a comparable loss on canonical images, leaving exact pair success unchanged or worse.
- **H3 — decoding artifact:** some or all of the apparent VCD gain is reproduced by the adaptive plausibility constraint (APC) or by greedy decoding without visual contrast.

The wording must describe a decoding-time change in sensitivity, not something the model "learns," because VCD does not train the model.

## 3. Scope

### Immediate progress-demo scope — due approximately September 15

Produce an honest, inspectable evidence package:

1. Git repository with a README and environment/config files.
2. Dataset inventory showing VLMBias domains, unique images, prompts, and duplicated resolutions.
3. A versioned manifest of at least 10 matched optical-illusion pairs already present in VLMBias.
4. One successful plain-LLaVA inference and one successful VCD inference on the same example.
5. Raw JSONL outputs plus a small parser audit table.
6. If time permits, a 10–25-pair baseline-versus-VCD smoke result. It must be labeled a smoke test, not a finding.

Environment setup, manifests, source code, and raw outputs count as real progress. Do not invent a result if model setup is incomplete.

### Research MVP — target October 2

- At least 50 manually validated pairs.
- One prompt form and one resolution per pair.
- Greedy, matched regular sampling, and full VCD.
- One seed for the first complete analysis; three seeds immediately afterward.
- Canonical accuracy, counterfactual accuracy, exact pair success, bias rate, and transition counts.

### Full study

- Reproduce the published aggregate regular/VCD comparison on the 2,784-row `main` split.
- Audit 50–100 high-quality pairs across optical illusions and at least one deterministic counting domain.
- Run the five decoding conditions in Section 6 with three seeds for stochastic conditions.
- Report paired uncertainty intervals, transition analysis, per-domain results, and qualitative examples.

### Stretch scope

- More than 100 validated pairs.
- A third domain.
- Alternate prompt/resolution robustness checks.
- A secondary general-perception benchmark.

Stretch work must not delay the paired MVP.

### Non-goals

- Training or fine-tuning a VLM.
- A broad leaderboard comparison of many decoding methods.
- Claiming that VLMBias accuracy alone proves grounding.
- Treating repeated prompts, resolutions, or seeds as independent data points.

## 4. Data and unit of analysis

The statistical unit is one underlying visual intervention, identified by `pair_id`. The two members are:

- `canonical`: the familiar configuration with target answer `canonical_gt`.
- `counterfactual`: the minimally modified configuration with target answer `counterfactual_gt`.

For the primary analysis, use Q1 and 768 px unless inspection shows a domain-specific reason to choose another fixed prompt or resolution. The choice must be frozen before examining comparative results.

### Pair-source priority

1. **Existing optical-illusion pairs:** VLMBias already contains original and modified versions with the same question and reversed answer. These are the fastest, safest MVP source.
2. **Deterministic programmatic domains:** game boards and chess pieces can be rendered from the same parameters with one controlled change. Prefer these for the counting extension.
3. **Flags:** include only after proving that the canonical and counterfactual render differ solely in the intended stars/stripes. The current public flag generator retrieves SVGs and uses an LLM to edit them, so uncontrolled changes are possible.
4. **Generated animals/logos:** exploratory only unless the original and edited images can be matched and independently validated.

### Pair acceptance criteria

A pair is included only if all of the following hold:

- The prompt is identical across both members.
- Image dimensions and rendering settings match.
- The intended intervention and both ground-truth answers are unambiguous.
- No unrelated visual change could plausibly alter the answer.
- A human review marks the pair `pass`.
- The manifest records provenance and enough parameters to recreate or identify both images.

Required pair-manifest fields:

```text
pair_id, domain, template_id, prompt, resolution,
canonical_path, counterfactual_path,
canonical_gt, counterfactual_gt, expected_bias,
intervention, generator_params, source_revision,
qc_status, qc_notes
```

## 5. Reproduction protocol

The aggregate reproduction and the paired audit are separate analyses.

### Aggregate reproduction

- Evaluate all 2,784 rows in VLMBias `main` with the exact dataset revision recorded.
- Run matched regular sampling and full VCD using the same model, prompt formatting, generation limits, sampling parameters, and seeds.
- Compare the result with the published LLaVA-1.5-7B figures of 16.92% regular and 19.18% VCD.
- Treat a difference larger than two percentage points as a discrepancy to investigate, not automatic proof of an implementation bug. Check dataset revision, prompt template, parser, checkpoint revision, and decoding configuration.

### Deduplicated analysis view

The 2,784 rows include repeated prompts and resolutions. Any inferential analysis must select one predeclared prompt and resolution per underlying image or cluster uncertainty by the underlying intervention. The full duplicated run may be retained only for a headline reproduction comparable with prior work.

## 6. Decoding conditions

The primary comparison is **regular sampling versus full VCD**. Greedy and the two ablations diagnose why a change occurs.

| Condition | Visual contrast | APC | Sampling | Runs per image |
|---|---:|---:|---:|---:|
| Greedy | No | No | No | 1 |
| Regular sampling | No | No | Yes | 3 seeds |
| APC only | No (`alpha = 0`) | Yes | Yes | 3 seeds |
| Contrast only | Yes | Disabled | Yes | 3 seeds |
| Full VCD | Yes | Yes | Yes | 3 seeds |

This produces 13 runs per image: greedy once plus four stochastic conditions at three seeds each.

Implementation notes:

- Recover and record the exact published generation parameters before claiming a reproduction.
- The official VCD code defaults to `cd_alpha=0.5` and `cd_beta=0.1`, but defaults must not be assumed to match the EnAR experiment.
- Implement an explicit APC on/off switch for the contrast-only condition rather than relying on a numerically fragile cutoff trick.
- Confirm that the regular baseline uses the same sampling configuration as VCD, without the distorted-image branch.
- Limit answers to a small fixed `max_new_tokens` value appropriate for the required `{number}` or `{Yes/No}` response.

## 7. Output contract and parsing

Write one append-only JSONL record per image, condition, and seed:

```text
run_id, pair_id, image_variant, domain, template_id,
model_revision, dataset_revision, prompt,
raw_output, parsed_answer, parse_status,
ground_truth, expected_bias, is_correct, is_bias_answer,
condition, seed, generation_config, config_hash,
git_sha, gpu_name, started_at, duration_ms
```

`parse_status` must be one of `valid`, `ambiguous`, or `invalid`. Never silently score an invalid parse as an ordinary wrong answer.

The parser must support curly-braced answers, bare digits, number words, and short sentences. Before reporting results:

- Unit-test known formats and malformed outputs.
- Hand-audit at least 30 real outputs and all ambiguous/invalid outputs in the MVP.
- Report the invalid rate by condition.

Do not save full-vocabulary logits for every token in the main run. For diagnostics, save top-k token scores or full scores only for a small, declared subset.

## 8. Metrics

For pair `i` and decoder `d`, let `C` and `CF` denote canonical and counterfactual correctness.

### Primary metric

```text
pair_success(d) = mean_i[correct(i, C, d) AND correct(i, CF, d)]
```

The primary effect is the paired difference in pair success between full VCD and matched regular sampling.

### Secondary metrics

- Canonical accuracy.
- Counterfactual accuracy.
- Balanced accuracy across the two image variants.
- Counterfactual bias rate: prediction equals the canonical/expected-bias answer.
- Exact directional discrimination: both answers are correct and change in the required direction.
- Image-response change rate: prediction differs between pair members; this is diagnostic, not sufficient evidence of grounding.
- Invalid parse rate.

### Transition metrics

Compare regular sampling with VCD on the same pair and seed:

- `CF correction`: regular is wrong and VCD is correct on the counterfactual.
- `CF regression`: regular is correct and VCD is wrong on the counterfactual.
- `C correction`: regular is wrong and VCD is correct on the canonical.
- `C regression`: regular is correct and VCD is wrong on the canonical.
- `pair improvement`: regular fails exact pair success and VCD achieves it.
- `tradeoff-only change`: the counterfactual is corrected while the canonical member regresses.

## 9. Interpretation rule

Freeze this rule before the full comparative analysis.

- **Evidence for improved grounding:** pair success increases, corrections are not explained mainly by canonical regressions, and the result is consistent across seeds and more than one template/domain.
- **Evidence for prior avoidance:** counterfactual accuracy rises while canonical accuracy falls and pair success is flat or worse, especially when tradeoff-only changes dominate.
- **Evidence for an APC/greedy artifact:** APC-only or greedy reproduces most of the full-VCD change without visual contrast.
- **Mixed/inconclusive:** effects differ by domain, uncertainty is wide, parser failures differ materially, or pair quality is questionable.

Do not equate a non-significant result with proof of no effect. Report the estimate and uncertainty.

## 10. Statistical analysis

- Report absolute percentage-point effects with 95% confidence intervals.
- Use paired bootstrap intervals over unique `pair_id`; cluster or resample by `template_id` when several pairs share a generator template.
- Use McNemar's test for binary correctness changes between matched decoding conditions, separately for canonical and counterfactual members. Use the exact version when discordant counts are small.
- Treat the full-VCD versus regular-sampling pair-success comparison as primary. Ablations and per-domain analyses are explanatory/secondary.
- Never count resolutions, prompt paraphrases, or random seeds as independent samples.

## 11. System design

Suggested repository layout:

```text
README.md
SPEC.md
environment.yml
configs/
  reproduction.yaml
  paired_mvp.yaml
data/
  manifests/
src/
  data/build_inventory.py
  data/build_pairs.py
  eval/run_inference.py
  eval/vcd_adapter.py
  eval/parse_answer.py
  analysis/summarize.py
  analysis/paired_stats.py
tests/
  test_parser.py
  test_pair_manifest.py
outputs/             # ignored; raw JSONL and tables
figures/             # selected tracked figures only
```

### Environment baseline

- Python 3.10 in an isolated Conda environment.
- Official VCD-compatible baseline: PyTorch 2.0.1, torchvision 0.15.2, and transformers 4.31.0.
- Use a CUDA 11.8 PyTorch build for the pinned PyTorch 2.0.1 stack; a CUDA 12.1 build is not published for that PyTorch version.
- Run batch size 1 in FP16 on one assigned RTX A5000. Additional GPUs are an optional throughput improvement, not a requirement.

The VCD monkeypatch changes Transformers generation internals. Pin every dependency and keep the VCD adapter isolated so failures are detectable.

## 12. Validation and acceptance tests

### Pipeline

- One canonical and one counterfactual image run end-to-end.
- The VCD branch is instrumented and asserts that a distorted image was processed.
- At least one diagnostic example has nonzero regular-versus-contrastive score differences. Final text is allowed to remain the same.
- Rerunning a fixed seed produces the same parsed answer and config hash.
- Interrupted jobs resume without duplicating completed records.

### Data

- Manifest paths resolve and pair IDs are unique.
- Canonical and counterfactual targets differ.
- No duplicate resolution/prompt variants enter the primary sample.
- Every primary pair passes manual QC.

### Analysis

- A tiny synthetic fixture with known transitions produces the expected metrics.
- Aggregate counts reconcile with the JSONL record count.
- Tables report denominators and invalid counts.

## 13. Revised execution plan

### September 12–15: progress evidence

1. Initialize the repository and add ignores before downloading data or weights.
2. Create the Python 3.10 environment with the pinned CUDA 11.8-compatible stack.
3. Snapshot source/model/dataset revisions.
4. Build the dataset inventory and first optical-pair manifest.
5. Run plain inference, then instrument and run VCD.
6. Save raw records, audit the parser, and produce one small table or pipeline-status screenshot.

### Week 2: reproduction

- Measure actual one-GPU throughput on 100 cases before estimating full-run duration.
- Run the complete main split under matched regular and VCD conditions.
- Investigate discrepancies and harden parsing.

### Week 3: paired data

- Complete existing optical pairs.
- Add a deterministic counting domain.
- Manually validate all pairs and freeze the manifest.

### Week 4: paired MVP

- Run 50 pairs under greedy, regular sampling, and full VCD.
- Produce the first pair-success and transition table.

### Weeks 5–8: ablations, statistics, robustness, packaging

- Add APC-only and contrast-only conditions.
- Expand only with validated pairs.
- Run paired statistics, figures, robustness checks, and final packaging.

## 14. Risks and fallbacks

| Risk | Trigger | Fallback |
|---|---|---|
| Old VCD stack fails | No VCD smoke inference after a bounded setup/debug effort | Report the reproduction issue and complete a benchmark/data paired audit; optionally port only the minimal VCD sampler with tests |
| Exact published score does not reproduce | Difference remains after prompt/config/parser/revision checks | Report the discrepancy transparently and use the internally matched regular baseline |
| Canonical pairs contain unrelated changes | QC failure or unclear provenance | Exclude them; use existing optical pairs and deterministic renderers |
| Counting domain has a floor effect | Both methods are near zero | Keep it as a finding but do not use it alone; retain optical pairs or add another clean domain |
| Parser differs by condition | Invalid/ambiguous rates diverge | Manually adjudicate blinded outputs and run parser-sensitivity analysis |
| GPU access is reduced | Only one A5000 is assigned | Run batch size 1 and schedule full jobs overnight; reduce pair count before reducing validation quality |

## 15. Deliverables

- Reproducible code and pinned environment.
- Source/model/dataset revision manifest.
- Validated paired-image manifest with QC notes.
- Raw prediction JSONL files.
- Parser tests and audit record.
- Aggregate reproduction table.
- Paired results table and transition matrix.
- Main figure: exact pair success by condition and domain.
- Short report explaining whether the result supports grounding, prior avoidance, an APC/greedy artifact, or a mixed conclusion.

## 16. Primary references

- [VLMBias paper](https://arxiv.org/abs/2505.23941) and [official code/data](https://github.com/anvo25/vlms-are-biased)
- [VCD paper](https://arxiv.org/abs/2311.16922) and [official implementation](https://github.com/DAMO-NLP-SG/VCD)
- [EnAR CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Envision_Attend_Then_Respond_Counterfactual_Hallucination_Mitigation_in_Large_Vision-Language_CVPR_2026_paper.html)
- [Yin et al., contrastive-decoding audit](https://arxiv.org/abs/2504.10020)
- [Bendre et al., reproducibility study](https://arxiv.org/abs/2607.25196)
