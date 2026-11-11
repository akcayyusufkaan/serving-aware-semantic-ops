# Palimpzest probe

Checks how Palimpzest 1.5.3 plans and executes a small pipeline (filter → map → map over six reviews)
when its models are two self-hosted vLLM servers. The installed package is not modified; the
one-vLLM-model-per-run check is lifted at runtime with `PATCH_GUARD=1`.

| File | Purpose |
| --- | --- |
| `pz_probe.py` | Runs one scenario and prints the chosen plan, sample outputs and per-server vLLM counters |
| `mock_vllm.py` | GPU-free stand-in for vLLM that logs every request with the port it hit |
| `mock_summary.py` | Requests per port and port changes between consecutive requests, from the mock log |
| `start_two_vllm.sh` | Starts two small models on a 4 GB GPU, both awake |

Install Palimpzest in its own environment; its dependency pins can conflict with vLLM's.

```bash
python3 -m venv ~/pz_venv && ~/pz_venv/bin/pip install palimpzest==1.5.3
```

## Without a GPU

```bash
python3 mock_vllm.py "$PWD/requests.jsonl" 8001 8002 &
PATCH_GUARD=1 ~/pz_venv/bin/python pz_probe.py split_mixed
python3 mock_summary.py requests.jsonl
```

## With real vLLM

```bash
./start_two_vllm.sh                      # in the vLLM environment
export PZ_MODEL_A=Qwen/Qwen2.5-0.5B-Instruct PZ_MODEL_B=HuggingFaceTB/SmolLM2-360M-Instruct
PATCH_GUARD=1 ~/pz_venv/bin/python pz_probe.py two_models_default
PATCH_GUARD=1 ~/pz_venv/bin/python pz_probe.py two_models_none_pipe
PATCH_GUARD=1 ~/pz_venv/bin/python pz_probe.py split_mixed
```

Small models may break Palimpzest's answer format; `None` fields then point to formatting, not to Palimpzest.

## Findings so far (mock vLLM)

- With the check lifted, two vLLM models run in the same query.
- Without a validator the plan always uses a single model, picked from Palimpzest's built-in
  MMLU-Pro table (`MaxQuality` → Qwen, `MinTime` → Llama).
- With a validator, sentinel sampling picks a different model per operator (a mixed plan).
- `pipelined` executes record by record (the harness's `naive`); `sequential` and `parallel`
  execute operator by operator (`batched`).
- During sentinel sampling consecutive requests switched server 17 times in 32 requests, so
  running Palimpzest against sleeping servers needs a proxy that wakes the target model.
- `sample_budget` larger than the number of possible samples hung the run (40 hung; 20 and 30 finished).
- `optimizer_strategy="none"` does not restrict the run to one model.
- Import fails without network access to the S3 model metadata (`PZ_OFFLINE=1` works around it).
- The default validator calls OpenAI o4-mini; a self-hosted setup needs its own validator.
