# outputs/

Raw predictions, gitignored. Each run dir has run_config.json, records.jsonl (one line per
generation) and run.log. Re-run the commands in the top-level README to regenerate; re-running a run
name resumes and doesn't duplicate anything.

- step5_plain_llava/ - one Muller-Lyer pair, greedy, plain LLaVA (2 records)
- step6_vcd_check/ - same pair, regular vs vcd, seeds 42/43, with first-step diagnostics (8)
- smoke_optical_22pairs/ - 22 pairs x both images x {greedy, regular, vcd} x seeds 42/43/44 (308)
- smoke_optical_22pairs_SUPERSEDED_perrun_reseed/ - same but with the seeding bug, kept for reference,
  don't use

Record fields: run_id, pair_id, image_variant, domain, sub_domain, template_id, model_revision,
dataset_revision, image_path, prompt, question, raw_output, parsed_answer, parse_status, parse_method,
ground_truth, expected_bias, is_correct, is_bias_answer, condition, seed, item_seed, generation_config,
config_hash, n_new_tokens, hit_max_new_tokens, n_cd_forward_calls, first_step_topk,
first_step_n_unmasked, git_sha, gpu_name, hostname, started_at, duration_ms (+ diagnostics with --diagnostics)
