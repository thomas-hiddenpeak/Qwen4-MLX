# S3 routed-expert patterns

`route-cases.json` contains six actual ordered top-10 expert-ID patterns from the original 11,057-token prompt / 128-output D2 routing capture. Its provenance, first-match selection predicates and source report hashes are included. It contains no captured activations or routing scores, and the six selected patterns are not a frequency-weighted workload sample.

The grouped gate/up operator probe uses these IDs with independent synthetic BF16 inputs and the ordinary router's rank-slot scores. It tests a forced-routing operator, not a replay of the source generation. See [the experiment result](../../docs/MTP_GROUPED_GATEUP_EXPERIMENT.md).
