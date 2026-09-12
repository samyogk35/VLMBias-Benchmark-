# Parser audit

318 records from outputs/smoke_optical_22pairs/records.jsonl, outputs/step5_plain_llava/records.jsonl, outputs/step6_vcd_check/records.jsonl

## distinct raw outputs (re-parsed with the current parser)

| n | raw_output | parsed | status | method |
|---:|---|---|---|---|
| 180 | `'No'` | No | valid | bare |
| 136 | `'Yes'` | Yes | valid | bare |
| 1 | `'No.'` | No | valid | bare |
| 1 | `'No, the circles are not equal in size. The larger circle is in the'` | No | valid | bare |

## parse status by condition

| condition | valid | ambiguous | invalid | not-valid rate |
|---|---:|---:|---:|---:|
| greedy | 46 | 0 | 0 | 0.0% |
| regular | 136 | 0 | 0 | 0.0% |
| vcd | 136 | 0 | 0 | 0.0% |

records that hit max_new_tokens: 1

## non-valid records to check by hand (0)


stored vs re-parsed mismatches: 0
