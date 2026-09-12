# outputs/ (git-ignored raw predictions)

Each run directory holds `run_config.json` (full config + `config_hash`), `records.jsonl` (one
append-only record per generation, SPEC §7 contract), and `run.log`. Regenerate with the commands in the
top-level README; re-running a run name resumes and never duplicates a `run_id`.

| run | what | records |
|---|---|---:|
| `step5_plain_llava/` | one Müller-Lyer pair, greedy, plain LLaVA (no VCD kwargs) | 2 |
| `step6_vcd_check/` | same pair, regular sampling vs. full VCD, seeds 42/43, with first-step diagnostics | 8 |
| `smoke_optical_22pairs/` | 22 smoke-subset pairs x {canonical, counterfactual} x {greedy, regular, vcd} x seeds {42,43,44} | 308 |
| `smoke_optical_22pairs_SUPERSEDED_perrun_reseed/` | same design but with `set_seed(seed)` before every generation — kept as evidence of the seeding flaw described in the README; do not analyse | 308 |

Record fields: `run_id, pair_id, image_variant, domain, sub_domain, template_id, model_revision, dataset_revision,
image_path, prompt (full conversation template), question (VLMBias prompt), raw_output, parsed_answer, parse_status,
parse_method, ground_truth, expected_bias, is_correct, is_bias_answer, condition, seed, item_seed, generation_config,
config_hash, n_new_tokens, hit_max_new_tokens, n_cd_forward_calls, first_step_topk, first_step_n_unmasked, git_sha,
gpu_name, hostname, started_at, duration_ms` (+ `diagnostics` when run with `--diagnostics`).
