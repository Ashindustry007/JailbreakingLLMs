# PAIR Together Variant Summary

## Table 2-style Metrics
- `gemma-3n-e4b-it`: JB% 100.0 (50/50), Queries/Success 5.12
- `llama-3-8b-instruct-lite`: JB% 90.0 (45/50), Queries/Success 15.87

## Table 4-style Runtime/Cost Estimates
- `gemma-3n-e4b-it`: wall 757.27s total, estimated output-cap cost $0.0865
- `llama-3-8b-instruct-lite`: wall 42146.65s total, estimated output-cap cost $0.3969

## Notes
- Figure 4 heatmap is generated from `category` and `jailbroken` fields in status files.
- Figure 5 is a log-parsed approximation from per-iteration summary lines, not the original paper's full K=1..12 ablation unless those runs are produced.
- Cost estimates use configured max output token caps and public price assumptions; they are not exact provider invoices.
