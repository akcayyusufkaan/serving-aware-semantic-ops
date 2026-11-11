# Serving-Aware Semantic Operators

Do semantic query plans chosen under API pricing stay optimal when the same models run on a
self-hosted vLLM server?

Semantic operator systems such as [Palimpzest](https://github.com/mitdbg/palimpzest) (with its
optimizer Abacus) and [LOTUS](https://github.com/guestrin-lab/lotus) choose a physical plan by
estimating cost as tokens × price, summed over operators. On your own GPU the bill is GPU time
instead, and the GPU time for the same tokens depends on prefix cache hits, batching and which
model is resident in memory. This repository holds the harness that measures how far the two
cost views diverge.

Senior project, CMPE 491/492, Boğaziçi University. Advisor: Prof. Atay Özgövde.

## Results so far

First measurements on one NVIDIA L4 (vLLM 0.30.0) with Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct
sharing the GPU through sleep mode (`configs/smoke_8b_l4.yaml`: six short reviews, two operators,
mean of 3 repeats; repeats differ by less than 1%):

| Plan | naive (s) | batched (s) | Ratio | Switches (naive / batched) | Switch time, naive | Cached prompt tokens (naive / batched) |
| --- | --- | --- | --- | --- | --- | --- |
| `all_llama` | 4.92 | 0.99 | 5.0× | 0 / 0 | – | 336 / 336 |
| `all_qwen` | 4.28 | 0.91 | 4.7× | 0 / 0 | – | 336 / 336 |
| `mixed` | 37.27 | 3.91 | 9.5× | 11 / 1 | 33.3 s (89%) | 0 / 320 |
| `moa` | 39.76 | 6.85 | 5.8× | 12 / 2 | 36.4 s (91%) | 80 / 480 |

- A plan uses the same number of tokens under both strategies, so its API cost is identical, while
  its GPU time differs by up to 9.5×.
- One model switch (sleep one 8B model, wake the other) takes 3.03 s; with 0.5B models it took 0.4 s.
- Switching also discards the prefix cache: the sleeping server's KV cache is dropped.
- Keeping a second model asleep is not free: it still holds about 1.8 GiB of GPU memory, which
  shrinks the active model's KV cache, and its weights occupy about 15 GB of host RAM.

The workload is a small test, so these numbers show the direction of the effects, not the size of
the effect on CUAD.

The full progress report, including related work and next steps, is in [docs/progress_report.md](docs/progress_report.md).

## How it works

One image, one config, one command. Only the launcher changes between environments.

```
launch/*  ──►  container (one Docker image)
               ├── runner: reads the config, executes the run matrix
               ├── vLLM server, model A   (awake)
               └── vLLM server, model B   (asleep, weights in CPU memory)
               mounts: configs/, results/, model cache
```

- **Image:** the official vLLM image plus the harness. Weights and tokens are not baked in.
- **Config:** models, plans, execution strategies, repeats and vLLM settings in one YAML file.
- **Runner:** starts one vLLM server per model and keeps only one awake (`/sleep`, `/wake_up`),
  so two 8B models share a 24 GB L4. Each run starts from the same state (first model awake,
  empty prefix cache), and the result is appended to `runs.jsonl` as soon as it finishes.
- **Resume:** every run has a deterministic `run_id`, so re-running the same command skips
  finished runs. A spot preemption or a Slurm time limit costs at most one run.

## Experiment

| | |
| --- | --- |
| Workload | CUAD, 100 contracts, two operators (does the clause exist, extract it) |
| Models | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct on one L4 (GCP g2-standard-16, 64 GB RAM) |
| Plans | `all_llama`, `all_qwen`, `mixed` (model per operator), `moa` (two proposers, one aggregator) |
| Strategies | `naive`: record by record, switching models when a step needs it · `batched`: operator by operator, each model woken once |
| Repeats | 3 |

Single-model plans never switch models, so their `naive`/`batched` gap isolates batching and
caching; multi-model plans add the switch cost on top. Each plan is then priced twice: tokens ×
API list price for the same open models, and measured GPU time × hourly GPU price. The two
rankings are compared with Kendall's tau and two Pareto frontiers.

## Repository layout

| Path | Contents |
| --- | --- |
| `harness/config.py` | Config loading, run matrix, `run_id`s |
| `harness/servers.py` | vLLM processes, sleep/wake, switch timing |
| `harness/metrics.py` | Prefix cache counters from `/metrics`, per-run deltas |
| `harness/plans.py` | Plan execution under `naive` or `batched` |
| `harness/workloads.py` | Workload loading and quality evaluation |
| `harness/runner.py` | Entry point, resume and result writing |
| `configs/smoke.yaml` | Two small models, for a 4 GB laptop GPU |
| `configs/smoke_l4.yaml` | The same small models on an L4 |
| `configs/smoke_8b_l4.yaml` | The same test with the two 8B models of the main experiment |
| `configs/cuad.yaml` | Main experiment |
| `launch/` | Docker (laptop); GCP network setup, VM creation, startup script and CPU test; Slurm + Apptainer (TRUBA); Kubernetes Job |
| `cloudbuild.*.yaml` | Cloud Build jobs: image build, VM creation with retries, CPU test image |
| `Dockerfile.cpu`, `tests/stub_vllm/` | CPU image with a stub vLLM for testing without a GPU |
| `experiments/palimpzest_probe/` | How Palimpzest plans over two self-hosted models |

## Running

### Without Docker

In an environment where vLLM is installed:

```bash
pip install -r requirements.txt
python3 -m harness.runner --config configs/smoke.yaml --output-dir ./results
```

`results/smoke/runs.jsonl` should contain 8 lines (4 plans × 2 strategies). Running the command
again prints `8 runs done, 0 remaining`.

### Docker

Requires the NVIDIA Container Toolkit.

```bash
./launch/local.sh configs/smoke.yaml
```

### GCP

Everything runs from Cloud Shell in the repository root. One-time setup:

```bash
PROJECT=<project-id> ./launch/gcp_cpu_test.sh                  # Artifact Registry, bucket, CPU test VM
PROJECT=<project-id> REGION=us-central1 ./launch/gcp_network.sh # per region: Private Google Access + Cloud NAT
```

The Hugging Face token goes into Secret Manager as `hf-token` (Llama is a gated model). VMs have
no external IP, so every region a VM may land in needs `gcp_network.sh` once.

A GPU run:

```bash
PROJECT=<project-id> CONFIG=configs/smoke_8b_l4.yaml MACHINE=g2-standard-16 RETRY=1 \
  ./launch/gcp_gpu_run.sh
```

- The config is uploaded to GCS, so a config change needs no image rebuild.
- Without `SKIP_BUILD=1`, Cloud Build builds the vLLM image first; with it, the image tagged with
  the current commit (or `TAG=<tag>`) is reused.
- `RETRY=1` leaves zone selection to Cloud Build, which tries every zone in `ZONES` every 10 minutes
  for up to 3 hours. L4s are often out of stock in a zone, so give several regions.
- The VM (`launch/gcp_vm_startup.sh`) runs the harness, uploads results plus the harness log, a
  memory trace (`*_mem.log`) and kernel messages (`*_system.log`), then deletes itself. It is
  stopped after `MAX_RUN` (default 3h) in any case.
- `SPOT=1` is about 40% cheaper but can be preempted within minutes when capacity is tight.

Constraints found on the L4 with two 8B models:

- `gpu_memory_utilization` must leave room for the sleeping server (about 1.8 GiB): 0.8 leaves no KV
  cache for Llama, 0.9 does not fit next to the sleeping model, 0.85 works.
- Host RAM: runs failed on 32 GB and 48 GB machines when the second model went to sleep, and
  succeed on `g2-standard-16` (64 GB). The exact cause on the smaller machines is not proven.

### TRUBA / Kubernetes

See `launch/truba.sbatch` and `launch/k8s_job.yaml`.

## Output

One JSON line per run in `results/<experiment>/runs.jsonl`:

| Field | Meaning |
| --- | --- |
| `run_id`, `plan`, `strategy`, `repeat` | Which run |
| `env`, `gpu_name`, `vllm_version`, `image_digest`, `config_hash` | Where and with what it ran |
| `wall_time_s` | Total run time |
| `n_model_switches`, `switch_time_s` | Sleep/wake count and time |
| `usage` | Prompt, completion and cached tokens per model |
| `prefix_cache` | Prefix cache queries and hits per server during the run |
| `quality_f1` | Quality against labels |

## Status

- [x] Harness, configs and launchers; tested against a stub vLLM (run matrix, switch counts, resume)
- [x] Palimpzest probe against a stub vLLM
- [x] GCP pipeline check on a CPU VM (`launch/gcp_cpu_test.sh`)
- [x] Smoke tests on an L4 with 0.5B and 8B models (two servers sharing one GPU via sleep/wake)
- [ ] Framework choice (Palimpzest or LOTUS) and the plan layer on top of it
- [ ] CUAD subset, prompts and F1
- [ ] CUAD runs on GCP and analysis

## References

- Liu et al., [Optimizing LLM Queries in Relational Data Analytics Workloads](https://arxiv.org/abs/2403.05821), MLSys 2025
- Russo et al., [Abacus: A Cost-Based Optimizer for Semantic Operator Systems](https://arxiv.org/abs/2505.14661)
- [vLLM](https://github.com/vllm-project/vllm)
