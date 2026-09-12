# Example raw records

Records where regular sampling and full VCD disagree on the same image and seed (from `outputs/smoke_optical_22pairs/records.jsonl`).

```json
{
 "run_id": "ab5963a5d77464c6",
 "pair_id": "opt_Ebbinghaus_strneg3_diffneg0p7",
 "image_variant": "counterfactual",
 "condition": "regular",
 "seed": 44,
 "item_seed": 693786856,
 "question": "Are the two red circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "No.",
 "parsed_answer": "No",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": true,
 "is_bias_answer": false,
 "n_cd_forward_calls": 0,
 "first_step_topk": [
  [
   "▁No",
   23.890625
  ],
  [
   "▁Yes",
   22.90625
  ],
  [
   "▁{",
   18.890625
  ],
  [
   "▁The",
   14.015625
  ],
  [
   "▁[",
   13.75
  ],
  [
   "▁There",
   13.0703125
  ],
  [
   "▁In",
   13.0234375
  ],
  [
   "▁Not",
   12.7890625
  ],
  [
   "▁It",
   12.609375
  ],
  [
   "No",
   12.125
  ]
 ],
 "first_step_n_unmasked": 32000,
 "duration_ms": 240.2,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```
```json
{
 "run_id": "279c274cf9c92da1",
 "pair_id": "opt_Ebbinghaus_strneg3_diffneg0p7",
 "image_variant": "counterfactual",
 "condition": "vcd",
 "seed": 44,
 "item_seed": 693786856,
 "question": "Are the two red circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "No",
 "parsed_answer": "No",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": true,
 "is_bias_answer": false,
 "n_cd_forward_calls": 2,
 "first_step_topk": [
  [
   "▁No",
   24.265625
  ],
  [
   "▁Yes",
   23.203125
  ],
  [
   "<unk>",
   -1.0000000150474662e+30
  ],
  [
   "<s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x03>",
   -1.0000000150474662e+30
  ],
  [
   "<0x04>",
   -1.0000000150474662e+30
  ],
  [
   "<0x02>",
   -1.0000000150474662e+30
  ],
  [
   "<0x01>",
   -1.0000000150474662e+30
  ],
  [
   "</s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x00>",
   -1.0000000150474662e+30
  ]
 ],
 "first_step_n_unmasked": 2,
 "duration_ms": 422.1,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```

```json
{
 "run_id": "e4344f3621ced7e4",
 "pair_id": "opt_Ebbinghaus_strneg5_diffneg0p7",
 "image_variant": "counterfactual",
 "condition": "regular",
 "seed": 44,
 "item_seed": 495127591,
 "question": "Are the two red circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "No, the circles are not equal in size. The larger circle is in the",
 "parsed_answer": "No",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": true,
 "is_bias_answer": false,
 "n_cd_forward_calls": 0,
 "first_step_topk": [
  [
   "▁No",
   23.875
  ],
  [
   "▁Yes",
   22.78125
  ],
  [
   "▁{",
   18.6875
  ],
  [
   "▁The",
   14.0078125
  ],
  [
   "▁[",
   13.6640625
  ],
  [
   "▁In",
   13.046875
  ],
  [
   "▁There",
   12.953125
  ],
  [
   "▁Not",
   12.703125
  ],
  [
   "▁It",
   12.515625
  ],
  [
   "No",
   12.140625
  ]
 ],
 "first_step_n_unmasked": 32000,
 "duration_ms": 594.9,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```
```json
{
 "run_id": "7267af0c9c60a45e",
 "pair_id": "opt_Ebbinghaus_strneg5_diffneg0p7",
 "image_variant": "counterfactual",
 "condition": "vcd",
 "seed": 44,
 "item_seed": 495127591,
 "question": "Are the two red circles equal in size? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "No",
 "parsed_answer": "No",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": true,
 "is_bias_answer": false,
 "n_cd_forward_calls": 2,
 "first_step_topk": [
  [
   "▁No",
   24.390625
  ],
  [
   "▁Yes",
   22.9375
  ],
  [
   "<unk>",
   -1.0000000150474662e+30
  ],
  [
   "<s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x03>",
   -1.0000000150474662e+30
  ],
  [
   "<0x04>",
   -1.0000000150474662e+30
  ],
  [
   "<0x02>",
   -1.0000000150474662e+30
  ],
  [
   "<0x01>",
   -1.0000000150474662e+30
  ],
  [
   "</s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x00>",
   -1.0000000150474662e+30
  ]
 ],
 "first_step_n_unmasked": 2,
 "duration_ms": 426.8,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```

```json
{
 "run_id": "d1a76b58218d4d9e",
 "pair_id": "opt_MullerLyer_str30_diff0p5",
 "image_variant": "counterfactual",
 "condition": "regular",
 "seed": 42,
 "item_seed": 907834306,
 "question": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "Yes",
 "parsed_answer": "Yes",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": false,
 "is_bias_answer": true,
 "n_cd_forward_calls": 0,
 "first_step_topk": [
  [
   "▁No",
   23.625
  ],
  [
   "▁Yes",
   23.03125
  ],
  [
   "▁{",
   18.65625
  ],
  [
   "▁The",
   13.2890625
  ],
  [
   "▁[",
   13.1171875
  ],
  [
   "▁There",
   12.4765625
  ],
  [
   "▁In",
   12.3671875
  ],
  [
   "▁Not",
   12.3046875
  ],
  [
   "▁It",
   12.2421875
  ],
  [
   "No",
   11.8671875
  ]
 ],
 "first_step_n_unmasked": 32000,
 "duration_ms": 213.3,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```
```json
{
 "run_id": "8ffb8e9ef1debbe0",
 "pair_id": "opt_MullerLyer_str30_diff0p5",
 "image_variant": "counterfactual",
 "condition": "vcd",
 "seed": 42,
 "item_seed": 907834306,
 "question": "Are the two horizontal lines equal in length? Answer in curly brackets, e.g., {Yes} or {No}.",
 "raw_output": "No",
 "parsed_answer": "No",
 "parse_status": "valid",
 "ground_truth": "No",
 "expected_bias": "Yes",
 "is_correct": true,
 "is_bias_answer": false,
 "n_cd_forward_calls": 2,
 "first_step_topk": [
  [
   "▁No",
   23.84375
  ],
  [
   "▁Yes",
   22.984375
  ],
  [
   "<unk>",
   -1.0000000150474662e+30
  ],
  [
   "<s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x03>",
   -1.0000000150474662e+30
  ],
  [
   "<0x04>",
   -1.0000000150474662e+30
  ],
  [
   "<0x02>",
   -1.0000000150474662e+30
  ],
  [
   "<0x01>",
   -1.0000000150474662e+30
  ],
  [
   "</s>",
   -1.0000000150474662e+30
  ],
  [
   "<0x00>",
   -1.0000000150474662e+30
  ]
 ],
 "first_step_n_unmasked": 2,
 "duration_ms": 426.8,
 "config_hash": "8275f879f8cf",
 "gpu_name": "NVIDIA RTX A5000"
}
```

## First-step diagnostics (step 6 check, `outputs/step6_vcd_check/records.jsonl`)

Clean vs. noised next-token logits and the VCD combination for the Müller-Lyer pair `opt_MullerLyer_str30_diff0p5`, computed outside the generation loop.

| variant | seed | cd forward calls | APC tokens kept | clean top-2 | noised top-2 | vcd top-2 | logit L1 clean-noised | prob L1 clean-vcd |
|---|---:|---:|---:|---|---|---|---:|---:|
| counterfactual | 42 | 2 | 2 | [['▁No', 23.625], ['▁Yes', 23.031]] | [['▁No', 23.359], ['▁Yes', 22.969]] | [['▁No', 23.891], ['▁Yes', 23.094]] | 2180 | 0.096 |
| counterfactual | 43 | 2 | 2 | [['▁No', 23.625], ['▁Yes', 23.031]] | [['▁No', 23.312], ['▁Yes', 23.0]] | [['▁No', 23.938], ['▁Yes', 23.062]] | 2246 | 0.129 |
| canonical | 42 | 2 | 2 | [['▁No', 23.453], ['▁Yes', 23.188]] | [['▁No', 23.344], ['▁Yes', 23.141]] | [['▁No', 23.562], ['▁Yes', 23.234]] | 1086 | 0.037 |
| canonical | 43 | 2 | 2 | [['▁No', 23.453], ['▁Yes', 23.188]] | [['▁No', 23.312], ['▁Yes', 23.141]] | [['▁No', 23.594], ['▁Yes', 23.234]] | 1301 | 0.052 |
