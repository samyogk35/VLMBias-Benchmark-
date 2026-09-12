# Parser audit

Records: 318 from outputs/smoke_optical_22pairs/records.jsonl, outputs/step5_plain_llava/records.jsonl, outputs/step6_vcd_check/records.jsonl

## Distinct raw outputs

| n | raw_output | parsed | status | method |
|---:|---|---|---|---|
| 180 | `'No'` | No | valid | bare |
| 136 | `'Yes'` | Yes | valid | bare |
| 1 | `'No.'` | No | valid | bare |
| 1 | `'No, the circles are not equal in size. The larger circle is in the'` | No | valid | bare |

## Parse status by condition

| condition | valid | ambiguous | invalid | invalid+ambiguous rate |
|---|---:|---:|---:|---:|
| greedy | 46 | 0 | 0 | 0.0% |
| regular | 136 | 0 | 0 | 0.0% |
| vcd | 136 | 0 | 0 | 0.0% |

Records that hit max_new_tokens: 1

## Non-valid records for hand review (0)


Stored-vs-reparsed mismatches: 0
