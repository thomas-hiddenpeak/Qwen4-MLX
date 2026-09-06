# Synthetic agent tools workload

This is a wholly authored, synthetic offline fixture. It contains no real user records, copied repository documentation, executed tools, or model-generated answers. All expected values were written before any generation and checked directly against the source packet. Distinct catalog/project records provide the long context; no paragraph is repeated as padding.

Frozen input: **11216 tokens**, system + user, text-only no-thinking chat framing. The canonical compact expected response is **53 tokens** before EOS. Run both **128 and 256** output budgets; natural EOS is allowed and actual output length must be reported. A valid JSON answer is not proof of AR/MTP token identity or a performance gain.

Use `prompt-token-ids.json` with `generate-gpu --tokens-file`, context 16384, and the desired budget. `source-data.json` is the complete synthetic packet. `expected.json` and `response-schema.json` are frozen functional contracts, not outputs sampled from a model. `manifest.json` records file/tokenizer/checker hashes and dispersed answer anchors. Tokenization was performed using the existing CPU-only `ane-runner tokenize` command, not by loading checkpoint weights.

From the repository root:

```sh
python3 scripts/check_agent_workload.py --fixture fixtures/gpu-agent-tools-11k
python3 scripts/check_agent_workload.py --fixture fixtures/gpu-agent-tools-11k \
  --generation results/your-run/generation.json
python3 scripts/check_agent_workload.py --fixture fixtures/gpu-agent-tools-11k \
  --retokenize --runner .build/release/ane-runner \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream
```

The checker rejects extra keys/text, duplicate JSON keys, non-finite constants, wrong scalar types, wrong answers and malformed JSON. Object key order and whitespace do not matter. The facts sources array must retain the requested sorted order. Each generation trial stores the full `prompt_tokens` integer array; the checker compares every ID against the frozen input. Integrity-only success checks no model output; this is reported explicitly. The manifest records the checker correction from an initial mistaken prompt-length assumption; no prompt, answer or input IDs changed.
