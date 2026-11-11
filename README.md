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
| Models | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct on one L4 (GCP g2-standard-8) |
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
| `configs/cuad.yaml` | Main experiment |
| `launch/` | Docker (laptop), GCE startup script and CPU test VM, Slurm + Apptainer (TRUBA), Kubernetes Job |
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

`launch/gcp_vm_startup.sh` is a GCE startup script that reads the image, bucket and config from
instance metadata, pulls the image from Artifact Registry, restores earlier results from GCS, runs
the harness (on the GPU when there is one), uploads the results and the harness log, and deletes
the VM. GPU VMs also get a hard `--max-run-duration`.

VMs have no external IP. Run the one-time network setup first (Private Google Access and Cloud NAT):

```bash
PROJECT=<project-id> ./launch/gcp_network.sh
```

To check that path without a GPU, run this from Cloud Shell in the repository root:

```bash
PROJECT=<project-id> ./launch/gcp_cpu_test.sh
```

It creates the Artifact Registry repository and the bucket, pushes a CPU image with a stub vLLM
(`Dockerfile.cpu`) and starts an e2-standard-2 VM that runs `configs/smoke.yaml`. Store the Hugging
Face token as the Secret Manager secret `hf-token` before the GPU run.

The GPU run builds the vLLM image on Cloud Build and starts a g2-standard-8 (1x L4) VM from a
Deep Learning VM image:

```bash
PROJECT=<project-id> ./launch/gcp_gpu_run.sh                            # configs/smoke_l4.yaml
PROJECT=<project-id> CONFIG=configs/cuad.yaml SPOT=1 ./launch/gcp_gpu_run.sh
```

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
- [ ] GCP pipeline check on a CPU VM (`launch/gcp_cpu_test.sh`)
- [ ] Smoke test on a real GPU (two servers sharing one GPU via sleep/wake)
- [ ] Framework choice (Palimpzest or LOTUS) and the plan layer on top of it
- [ ] CUAD subset, prompts and F1
- [ ] CUAD runs on GCP and analysis

## References

- Liu et al., [Optimizing LLM Queries in Relational Data Analytics Workloads](https://arxiv.org/abs/2403.05821), MLSys 2025
- Russo et al., [Abacus: A Cost-Based Optimizer for Semantic Operator Systems](https://arxiv.org/abs/2505.14661)
- [vLLM](https://github.com/vllm-project/vllm)
