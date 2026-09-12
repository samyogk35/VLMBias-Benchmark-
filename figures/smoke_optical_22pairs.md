# Smoke test results (not a finding)

Source: outputs/smoke_optical_22pairs/records.jsonl, 308 records, 22 pairs, seeds [0, 42, 43, 44], config_hash 8275f879f8cf, model rev 4481d270

Optical illusion pairs, Q1, 768px. Canonical GT = Yes (really equal), counterfactual GT = No.
Invalid/ambiguous parses count as wrong here and are also listed in any_invalid_pair_rate.

## by condition (all seeds pooled)

| condition   |   pairs |   pair_runs |   canonical_acc |   counterfactual_acc |   balanced_acc |   pair_success |   cf_bias_rate |   response_change_rate |   any_invalid_pair_rate |
|:------------|--------:|------------:|----------------:|---------------------:|---------------:|---------------:|---------------:|-----------------------:|------------------------:|
| greedy      |      22 |          22 |           0.455 |                0.545 |          0.500 |          0.000 |          0.455 |                  1.000 |                   0.000 |
| regular     |      22 |          66 |           0.470 |                0.621 |          0.545 |          0.258 |          0.379 |                  0.576 |                   0.000 |
| vcd         |      22 |          66 |           0.470 |                0.636 |          0.553 |          0.273 |          0.364 |                  0.561 |                   0.000 |

## by condition and seed

| condition   |   seed |   pairs |   pair_runs |   canonical_acc |   counterfactual_acc |   balanced_acc |   pair_success |   cf_bias_rate |   response_change_rate |   any_invalid_pair_rate |
|:------------|-------:|--------:|------------:|----------------:|---------------------:|---------------:|---------------:|---------------:|-----------------------:|------------------------:|
| greedy      |      0 |      22 |          22 |           0.455 |                0.545 |          0.500 |          0.000 |          0.455 |                  1.000 |                   0.000 |
| regular     |     42 |      22 |          22 |           0.545 |                0.591 |          0.568 |          0.273 |          0.409 |                  0.591 |                   0.000 |
| regular     |     43 |      22 |          22 |           0.455 |                0.545 |          0.500 |          0.273 |          0.455 |                  0.455 |                   0.000 |
| regular     |     44 |      22 |          22 |           0.409 |                0.727 |          0.568 |          0.227 |          0.273 |                  0.682 |                   0.000 |
| vcd         |     42 |      22 |          22 |           0.545 |                0.636 |          0.591 |          0.318 |          0.364 |                  0.545 |                   0.000 |
| vcd         |     43 |      22 |          22 |           0.455 |                0.545 |          0.500 |          0.273 |          0.455 |                  0.455 |                   0.000 |
| vcd         |     44 |      22 |          22 |           0.409 |                0.727 |          0.568 |          0.227 |          0.273 |                  0.682 |                   0.000 |

## by illusion type (all seeds pooled)

| condition   | sub_domain         |   pairs |   canonical_acc |   counterfactual_acc |   pair_success |
|:------------|:-------------------|--------:|----------------:|---------------------:|---------------:|
| greedy      | Ebbinghaus         |       4 |           0.000 |                1.000 |          0.000 |
| greedy      | MullerLyer         |       4 |           0.000 |                1.000 |          0.000 |
| greedy      | Poggendorff        |       6 |           1.000 |                0.000 |          0.000 |
| greedy      | Ponzo              |       2 |           0.000 |                1.000 |          0.000 |
| greedy      | VerticalHorizontal |       2 |           0.000 |                1.000 |          0.000 |
| greedy      | Zollner            |       4 |           1.000 |                0.000 |          0.000 |
| regular     | Ebbinghaus         |       4 |           0.250 |                0.833 |          0.250 |
| regular     | MullerLyer         |       4 |           0.333 |                0.500 |          0.083 |
| regular     | Poggendorff        |       6 |           0.500 |                0.667 |          0.278 |
| regular     | Ponzo              |       2 |           0.333 |                0.667 |          0.333 |
| regular     | VerticalHorizontal |       2 |           0.500 |                0.667 |          0.333 |
| regular     | Zollner            |       4 |           0.833 |                0.417 |          0.333 |
| vcd         | Ebbinghaus         |       4 |           0.250 |                0.833 |          0.250 |
| vcd         | MullerLyer         |       4 |           0.333 |                0.583 |          0.167 |
| vcd         | Poggendorff        |       6 |           0.500 |                0.667 |          0.278 |
| vcd         | Ponzo              |       2 |           0.333 |                0.667 |          0.333 |
| vcd         | VerticalHorizontal |       2 |           0.500 |                0.667 |          0.333 |
| vcd         | Zollner            |       4 |           0.833 |                0.417 |          0.333 |

## regular -> vcd, same pair and seed

| metric | count |
|---|---:|
| n_pair_seed | 66 |
| CF_correction | 1 |
| CF_regression | 0 |
| C_correction | 0 |
| C_regression | 0 |
| pair_improvement | 1 |
| pair_regression | 0 |
| tradeoff_only | 0 |

## greedy -> vcd (greedy is one run, compared against every vcd seed)

| metric | count |
|---|---:|
| n_pair_seed | 66 |
| CF_correction | 17 |
| CF_regression | 11 |
| C_correction | 12 |
| C_regression | 11 |
| pair_improvement | 18 |
| pair_regression | 0 |
| tradeoff_only | 8 |

## answer counts

|                               |   No |   Yes |
|:------------------------------|-----:|------:|
| ('greedy', 'canonical')       |   12 |    10 |
| ('greedy', 'counterfactual')  |   12 |    10 |
| ('regular', 'canonical')      |   35 |    31 |
| ('regular', 'counterfactual') |   41 |    25 |
| ('vcd', 'canonical')          |   35 |    31 |
| ('vcd', 'counterfactual')     |   42 |    24 |

## time per generation (ms)

| condition   |   count |   mean |   median |
|:------------|--------:|-------:|---------:|
| greedy      |      44 |    219 |      215 |
| regular     |     132 |    218 |      216 |
| vcd         |     132 |    431 |      431 |

