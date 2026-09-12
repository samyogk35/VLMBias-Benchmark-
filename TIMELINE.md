# Project Timeline — VCD Grounding Audit

**Start:** Tue Sep 8, 2026 · **Budget:** 10–20 hrs/week · *
**Assignment 3 progress report due:** ~Tue Sep 15, 2026
**MVP (first complete answer to the research question):** end of Week 4 — **Fri Oct 2**
**Full study packaged:** end of Week 8 — **Fri Oct 30**

---

## The single most important planning fact

**Running the experiments takes hours, not weeks.**

| Job | Generations | Wall time on 4 GPUs |
|---|---:|---:|
| Full VLMBias main split, regular + VCD | 5,568 | ~45 min |
| Full paired study (300 pairs × 2 imgs × 13 runs) | 7,800 | ~65 min |
| MVP subset (50 pairs × 2 × 3 runs) | 300 | ~5 min |

> **⚠️ GPU constraint (added Sep 12):** I cannot use all 4 GPUs on the machine — the node is shared, so only a subset is available to me. The wall times above assume 4 GPUs; scale them up accordingly (e.g. ~2× on 2 GPUs, ~4× on 1 GPU). Even at 1 GPU the full split is ~3 hrs and the MVP subset ~20 min, so the "hours, not weeks" conclusion still holds — but full runs should be launched overnight rather than mid-session.

So the schedule below does **not** budget weeks for "running experiments." It budgets them for the three things that actually consume time:

1. **Getting VCD's pinned stack to build** (it wants `transformers==4.31.0`; your base Python is 3.13, which will not work)
2. **Building and hand-validating the canonical–counterfactual pairs** — this is the actual research contribution and the critical path
3. **Answer parsing** — if `"There are four legs"` doesn't parse to `4`, every accuracy number you report is wrong, and it fails silently

---

## What one week buys you at 10–20 hrs

Use this to sanity-check any re-planning:

| Task type | Realistic time |
|---|---|
| Stand up a pinned ML env from scratch, including dependency fights | 1 week |
| Reverse-engineer one image generator + validate its output by hand | 1 week per domain |
| Write + debug an answer parser against real messy model output | 3–4 hrs |
| Run any experiment in this project | < 2 hrs |
| Statistics, figures, and writing them up | 1 week |

---

# WEEK 1 — Sep 8–13 · Environment + pilot
### Goal: have a real number to show in the Assignment 3 report

This is the week that produces your progress-report evidence. Day-level detail because it's the week you're in.

### Day 1 — Tue Sep 8 (3 hrs) · Repo + environment
- `git init` in `ss/`, add `.gitignore` (exclude weights, images, `*.jsonl` outputs), push to GitHub — **the report needs this link**
- `conda create -n vcd python=3.10` — 3.10, not 3.13; `transformers==4.31.0` will not build on 3.13
- Install: `torch` (cu121 build), `transformers==4.31.0`, `accelerate`, `sentencepiece`, `protobuf`
- **Gate:** `torch.cuda.is_available()` is True and reports the GPUs actually allocated to me (not all 4 — see GPU constraint above); set `CUDA_VISIBLE_DEVICES` to only those

### Day 2 — Wed Sep 9 (3 hrs) · Weights + data
- Download `liuhaotian/llava-v1.5-7b` (~14 GB, you have 1.8 TB free)
- Clone `DAMO-NLP-SG/VCD` and `anvo25/vlms-are-biased`; pull the HF dataset `anvo25/vlms-are-biased`
- Install the LLaVA package from the VCD repo's vendored copy
- **Gate:** one image in, one plain-LLaVA answer printed to console

### Day 3 — Thu Sep 10 (4 hrs) · VCD path + 100-case pilot
- Wire in VCD's two pieces: the sampling monkeypatch (`vcd_sample`) and the noised-image function (`vcd_add_noise`)
- Build a stratified 100-case pilot subset across domains
- Run both conditions, regular and VCD
- Log to JSONL from the start: `case_id, domain, prompt, raw_output, parsed_answer, ground_truth, seed, config_hash`
- **Gate:** both conditions complete and their outputs actually differ (if identical, the monkeypatch didn't attach — a classic silent VCD failure)

### Day 4 — Fri Sep 11 (4 hrs) · Parser + first numbers
- Write the answer parser: digits, number words, and answers wrapped in a sentence
- Hand-check the parser against 30 raw outputs — do not trust it unaudited
- Produce an accuracy table by domain, regular vs VCD
- Compare to the published EnAR figures (16.92% regular / 19.18% VCD)
- **Screenshot this table — it is your primary Assignment 3 evidence**

### Day 5 — Sat/Sun (3 hrs) · Write the report
- Record measured runtime per condition (feeds every later estimate)
- README with setup steps, commit everything, verify the repo is clean on a fresh clone
- Write Assignment 3 (scaffold at the bottom of this file)

**Week 1 exit gate:** Both conditions run reproducibly on 100 cases, with saved logits and a parser you've eyeballed. If VCD will not run by Sunday, stop and pivot to the fallback in your proposal — do not spend Week 2 rebuilding VCD.

---

# WEEK 2 — Sep 14–20 · Full reproduction + parser hardening
### Goal: your own version of the published number, on the full split

- Run the complete 2,784-row main split, regular + VCD (~45 min of compute)
- Deduplicate to one resolution and one prompt form per underlying case for the statistics; keep the full run for the headline comparison
- Harden the parser against every failure mode the full split exposes; track an explicit `invalid` rate rather than scoring unparseable output as wrong
- Write the experiment manifest: config hash, seed, git SHA recorded with every run
- **Gate:** your reproduced regular/VCD numbers are within a point or two of 16.92 / 19.18. If they aren't, you have a bug, and finding it now is much cheaper than in Week 6.

---

# WEEK 3 — Sep 21–27 · Pair construction (the hard part)
### Goal: canonical counterparts for one domain

This is the critical path. Everything downstream is blocked on it.

- Read the VLMBias generator scripts and find where the counterfactual edit is applied
- Start with **flags** — flat vector assets, most likely to expose a clean canonical setting
- Generate canonical counterparts at identical resolution, prompt, and rendering settings; the *only* difference is that the correct answer is the familiar one
- Build the versioned pair manifest: `pair_id, domain, canonical_path, counterfactual_path, canonical_gt, counterfactual_gt, generator_params`
- Hand-check every pair. All of them. A bad pair is an invisible wrong result.
- **Gate:** 50+ validated flag pairs. If the generator won't cleanly produce canonical versions for flags, switch domains immediately rather than fighting it.

---

# WEEK 4 — Sep 28–Oct 2 · **MVP**
### Goal: the first real answer to your research question

- Extend pairs to a second domain (chess pieces or game boards) — target 150+ pairs total
- Implement the condition splits: greedy, plain sampling, sampling + APC only, contrast-only without APC, full VCD
- **Run the MVP: 50 flag pairs × 3 conditions (greedy / sampling / full VCD) × 1 seed**
- Compute: counterfactual accuracy, canonical accuracy, **pair success**, and the corrections-vs-regressions transition counts

### This is your demo moment — Fri Oct 2

By the end of this week you can put a table in front of anyone showing whether VCD's counterfactual gain is paid for by breaking canonical answers. One domain, one seed, no confidence intervals — but it is a **complete, honest answer to the actual research question**, and the rest of the project is scaling it up and making it statistically defensible.

If you need to demo the project's potential to anyone, this is the artifact.

---

# WEEK 5 — Oct 5–11 · Full condition matrix
- All pairs, all 5 conditions, 3 fixed seeds (~1 hr compute)
- Freeze the protocol and the interpretation rule **before looking at results** — write the decision rule into the repo and commit it, so the timestamp proves you didn't set it after seeing the numbers
- Verify seed-to-seed stability

---

# WEEK 6 — Oct 12–18 · Statistics
- Paired differences with 95% bootstrap intervals, clustered by generator template (not by dataset row)
- McNemar tests for paired correctness changes
- Per-domain breakdowns — never one pooled "bias score"
- Pull the specific cases VCD fixed and the ones it broke; read them

---

# WEEK 7 — Oct 19–25 · Figures + robustness
- Main figure: pair-success by condition and domain
- Transition matrix: what changed to what
- Robustness: does the conclusion survive alternate parsing rules and dropping any single domain?
- Optional MME perception check, only if everything above is complete

---

# WEEK 8 — Oct 26–30 · Package
- Final report, code cleanup, environment lockfile
- Verify reproducibility from a fresh clone
- Data manifest, all raw predictions, presentation

---

## Where this plan can break, ranked

| Risk | When you'll know | What to do |
|---|---|---|
| VCD won't run under its pinned stack | Week 1, Day 3 | Fall back to the benchmark-only paired audit in your proposal |
| Generators won't expose clean canonical settings | Week 3 | Drop that domain, don't substitute internet images |
| Parser silently miscounts | Week 2 if you audit; Week 6 if you don't | Hand-check 30 raw outputs per domain, track invalid rate separately |
| Reproduction doesn't match published numbers | Week 2 | Debug now; a mismatch here invalidates everything downstream |
| Scope creep into a 3rd domain | Week 4–5 | Two domains with 150 clean pairs beats three domains with sloppy ones |

---

## Assignment 3 scaffold — fill in after Week 1

Do not write this until you have real numbers. Every bracket is a blank you fill from actual output.

**1. Project Overview**
Two or three sentences. The dog-with-five-legs framing, then the research question: does VCD's reported gain mean better seeing, or just avoiding familiar answers.

**2. Work Completed**
- Environment: conda env, Python 3.10, `transformers==4.31.0`, LLaVA-1.5-7B on 4× A5000
- Reproduced the VCD pipeline and confirmed the contrastive path is active
- Ran a stratified [N]-case pilot under both conditions
- Built the answer parser and hand-validated it on [N] raw outputs
- Measured throughput: [X] s/sample regular, [Y] s/sample VCD

**3. Evidence of Progress**
- GitHub link
- Screenshot of the accuracy table, regular vs VCD, by domain
- A code excerpt showing the VCD noise + contrastive step
- 2–3 raw JSONL records showing a case where the two conditions disagree

**4. Challenges**
Be specific and technical — this section is where graders look for real work. Likely honest content: the Python 3.13 / `transformers==4.31.0` incompatibility and how you resolved it; verifying the monkeypatch actually attached rather than silently no-op'ing; parser edge cases in raw model output.

**5. Next Steps**
Full-split reproduction and the comparison to the published 16.92 / 19.18 figures; then canonical pair construction for flags, targeting the MVP paired result by Oct 2.
