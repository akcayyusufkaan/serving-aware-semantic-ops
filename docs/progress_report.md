# Progress Report: Serving-Aware Cost Modeling for Semantic Operators

Yusuf Kaan Akçay · CMPE 491 · Advisor: Prof. Atay Özgövde · November 2026

## 1. Summary

Semantic operator optimizers such as Abacus (Palimpzest) choose plans by pricing each operator as
tokens × API price and summing over operators. This project measures whether those plans stay the
best ones when the same models run self-hosted on vLLM, where the cost is GPU time.

Status:

| Step | Status |
| --- | --- |
| Literature review | Done |
| vLLM measurement tooling, validated on a laptop GPU | Done |
| Portable experiment harness (one image for laptop, GCP, TRUBA, Kubernetes) | Done |
| Palimpzest compatibility probe (without GPU) | Done |
| First measurements on an NVIDIA L4 (0.5B and 8B models) | Done |
| CUAD workload, prompts and F1 | Next |
| Abacus plans executed through Palimpzest on vLLM | Next |
| Cost-ranking comparison (API vs GPU time) | Next |

The first L4 measurements already show the central effect: the same plan uses the same number of
tokens under two execution strategies, so its API cost is identical, while its GPU time differs by
up to 9.5× (Section 6).

## 2. Research question and hypotheses

Does a semantic query plan that is optimal under API pricing remain optimal when executed on a
self-hosted vLLM server, and if not, by how much do the two cost views diverge?

The two papers this builds on:

- **Liu et al., MLSys 2025** ([arXiv 2403.05821](https://arxiv.org/abs/2403.05821)): reordering rows
  and columns to increase prefix-cache reuse speeds up LLM queries on self-hosted vLLM by up to 3.4×
  without changing the number of tokens. A token-based cost estimate cannot see this difference.
- **Russo et al., Abacus** ([arXiv 2505.14661](https://arxiv.org/abs/2505.14661)): a cost-based
  optimizer for semantic operators that trades off quality, cost and latency. Operator cost is
  tokens × model price and plan cost is the sum over operators; all experiments use model APIs.
  On CUAD it chose Mixture-of-Agents in all ten trials.

| # | Hypothesis | How it is tested |
| --- | --- | --- |
| H1 | The self-hosted cost of multi-model plans (mixed, Mixture-of-Agents) depends on execution order; switching models per record can push them off the Pareto frontier. | Run the same plan record by record and operator by operator; compare GPU time and number of model switches. |
| H2 | Running operators that share a model back to back increases prefix-cache reuse, which an additive cost model ignores. | Measure prefix-cache hits and GPU time with the token count held fixed. |
| H3 | Plans ranked by API cost and by GPU cost are ordered differently. | Kendall's tau between the two rankings; compare the two Pareto frontiers. |

## 3. Related work and the gap

The citation trees of the two papers above were followed three levels deep on Semantic Scholar
(56 direct citations, about 2,170 papers in total), filtered by keyword, and the closest ~20 papers
were read. The search and screening were done with the help of Claude.

Systems that choose a plan do not know the serving state of the GPU (which model is loaded, what is
cached), and systems that execute plans efficiently on GPUs do not choose the plan. No work was found
that measures this gap for semantic operator optimizers.

| Work | Side | Relation to this project |
| --- | --- | --- |
| [Halo](https://arxiv.org/abs/2509.02121) (2025) | Execution | Cost model covers prefill/decode, cache sharing and model loading, but takes the model assignment of a plan as given. |
| [Kalypso](https://arxiv.org/html/2607.23815v1) (2026) | Execution | Shares KV cache across semantic operators; single model, no plan choice. |
| [Helium](https://arxiv.org/html/2603.16104v1) (2026) | Execution | Cache- and batch-aware scheduling; single model. |
| [MOSAIC](https://arxiv.org/abs/2606.03014) (2026) | Execution | Schedules Mixture-of-Agents on limited GPUs; no plan choice. |
| [Stretto](https://arxiv.org/html/2602.04430v2) (2026) | Plan choice | Models operator cost by measured time; no model switching, no API vs GPU comparison. |
| [BlendSQL v0.1.0](https://arxiv.org/abs/2606.31808) (2026) | Plan choice | Compares open models on a local GPU with APIs; rule-based optimizer. |
| [AI×DB / NeurEngine](https://arxiv.org/abs/2603.03772) (2026) | Vision | Names model placement outside the optimizer as an open problem. |
| [Akillioglu](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/PhD/PhD25_20.pdf) (VLDB 2025 PhD Workshop) | Vision | Proposes a cost model that includes KV-cache hits; proposal stage, SQL and a single model. |
| [Multi-Model LLM Schedulers](https://arxiv.org/html/2605.19593v1) (2026) | Model switching | Reports about 4–5 s to reload an 8B model; a warning that switch cost may be small relative to the workload. |

The contribution is not a new system but a measurement of this gap for semantic operator
optimizers, which current work describes as open.

## 4. Measurement tooling

Every quantity the experiment needs is exposed by vLLM. This was verified with vLLM 0.30.0 and
Qwen2.5-0.5B on a laptop GPU (RTX 3050 Ti, 4 GB):

| Quantity | Source | Check |
| --- | --- | --- |
| Cached prompt tokens per request | `usage.prompt_tokens_details.cached_tokens` (`--enable-prompt-tokens-details`) | Same prompt twice: 0, then high; 0 again after a cache reset |
| Server-wide cache reuse | `/metrics`: `prefix_cache_queries`, `prefix_cache_hits` | Consistent with the per-request values |
| Clean state between runs | `POST /reset_prefix_cache` | Works |
| Model switch cost | `POST /sleep`, `POST /wake_up` | Works (Section 6) |

Both LOTUS and Palimpzest assign a cost of 0 to self-hosted models, so their optimizers do not see
any cost on a self-hosted deployment.

## 5. Experiment harness

One Docker image, one YAML config and one command; only the launcher differs between a laptop, a
GCP VM, TRUBA (Slurm + Apptainer) and Kubernetes. The runner starts one vLLM server per model,
keeps exactly one awake with sleep mode, executes every (plan, strategy, repeat) combination from
the same starting state and appends one JSON line per run. Runs have deterministic IDs, so an
interrupted experiment resumes where it stopped. Details are in the [README](../README.md).

Experiment design:

| | |
| --- | --- |
| Plans | `all_llama`, `all_qwen`, `mixed` (model per operator), `moa` (two proposers, one aggregator) |
| Strategies | `naive`: record by record, switching models when a step needs it · `batched`: operator by operator, each model woken once |
| Held fixed | Config (hash stored with every run), GPU, vLLM version, image digest, temperature 0, empty prefix cache at the start of each run |
| Measured | Wall time, number and duration of model switches, tokens per model, prefix-cache hits, F1 |

Single-model plans never switch, so their naive/batched gap isolates batching and caching; for
multi-model plans the switch cost comes on top.

## 6. First results on an NVIDIA L4

Both runs use the same short test workload (six reviews, two operators), vLLM 0.30.0, and are the
mean of 3 repeats. Repeats differ by less than 1%. Source: `results/<experiment>/runs.jsonl` in the
project bucket.

Qwen2.5-0.5B and SmolLM2-360M (`configs/smoke_l4.yaml`):

| Plan | naive (s) | batched (s) | Switches (naive / batched) | Switch time, naive | Cached prompt tokens (naive / batched) |
| --- | --- | --- | --- | --- | --- |
| `all_a` | 0.50 | 0.14 | 0 / 0 | – | 336 / 336 |
| `all_b` | 0.49 | 0.20 | 0 / 0 | – | 336 / 336 |
| `mixed` | 5.32 | 0.61 | 11 / 1 | 4.74 s (89%) | 0 / 320 |
| `moa` | 5.40 | 0.92 | 12 / 2 | 4.99 s (92%) | 80 / 480 |

Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (`configs/smoke_8b_l4.yaml`):

| Plan | naive (s) | batched (s) | Ratio | Switches (naive / batched) | Switch time, naive | Cached prompt tokens (naive / batched) |
| --- | --- | --- | --- | --- | --- | --- |
| `all_llama` | 4.92 | 0.99 | 5.0× | 0 / 0 | – | 336 / 336 |
| `all_qwen` | 4.28 | 0.91 | 4.7× | 0 / 0 | – | 336 / 336 |
| `mixed` | 37.27 | 3.91 | 9.5× | 11 / 1 | 33.3 s (89%) | 0 / 320 |
| `moa` | 39.76 | 6.85 | 5.8× | 12 / 2 | 36.4 s (91%) | 80 / 480 |

Findings:

1. **Same tokens, different GPU time.** A plan uses the same number of tokens under both
   strategies (e.g. 610 for `mixed`), so its API cost is identical, while its GPU time differs by up
   to 9.5×.
2. **Model switches dominate multi-model plans.** One switch takes 3.03 s with 8B models (0.4 s
   with the small models). Under the naive strategy, switches account for 89–91% of the run time.
3. **Switching discards the prefix cache.** A sleeping server drops its KV cache, so `mixed` under
   the naive strategy gets no cached tokens at all, against 320 under the batched strategy. H1 and
   H2 are therefore linked.
4. **Keeping a second model asleep has a cost.** The sleeping 8B server still holds about 1.8 GiB of
   GPU memory, which shrinks the active model's KV cache, and its weights occupy about 15 GB of host
   RAM. API pricing sees none of this.
5. The per-request `cached_tokens` and the `/metrics` cache-hit counters agree in every run.

These numbers come from a small test workload and show the direction of the effects, not their size
on CUAD.

## 7. Palimpzest probe

Palimpzest 1.5.3 was tested against a stub vLLM server that logs every request
(`experiments/palimpzest_probe/`):

- With its one-vLLM-model-per-run check lifted at runtime, two self-hosted models work in one query.
- Without a validator the plan always uses a single model, chosen from Palimpzest's built-in
  MMLU-Pro table.
- With a validator, Abacus's sentinel sampling selects a different model per operator (a mixed plan).
- Its `pipelined` execution corresponds to the harness's naive strategy and `sequential`/`parallel`
  to the batched strategy.

Running Abacus plans self-hosted therefore needs four changes: lifting the one-model check, shipping
the model metadata Palimpzest otherwise downloads at import time, a validator based on CUAD labels
(the default one calls OpenAI), and a small proxy that wakes the right vLLM server for each request.

## 8. Problems encountered

- **GPU capacity:** L4s were frequently out of stock across GCP zones. VM creation now retries a list
  of zones in several regions automatically.
- **GPU memory with two 8B models:** `gpu_memory_utilization` must leave room for the sleeping
  server. 0.8 left no KV cache for Llama, 0.9 did not fit next to the sleeping model, 0.85 works.
- **Host memory:** the runs failed when the second model went to sleep on machines with 32 GB and
  48 GB of RAM and succeeded with 64 GB. The exact cause on the smaller machines is not proven yet;
  since host memory for sleeping models is part of the self-hosted cost, it will be investigated
  with memory profiling.

## 9. Next steps

1. Prepare the CUAD workload: a 100-contract subset, prompts and F1 against the labels. Contract
   length also decides how much KV cache is needed per request.
2. Execute the plans Abacus selects through Palimpzest on vLLM, with the four changes in Section 7.
3. Price every plan both ways (tokens × API list price, GPU time × hourly price) and compare the
   rankings with Kendall's tau and the two Pareto frontiers.
4. Profile host and GPU memory around model switches.
5. Decide whether single-model plans should run with only one server loaded, which gives them a
   larger KV cache and may be the fairer baseline.

## References

- Liu et al., [Optimizing LLM Queries in Relational Data Analytics Workloads](https://arxiv.org/abs/2403.05821), MLSys 2025
- Russo et al., [Abacus: A Cost-Based Optimizer for Semantic Operator Systems](https://arxiv.org/abs/2505.14661)
- [Palimpzest](https://github.com/mitdbg/palimpzest) · [LOTUS](https://github.com/guestrin-lab/lotus) · [vLLM](https://github.com/vllm-project/vllm) · [vLLM sleep mode](https://docs.vllm.ai/en/stable/features/sleep_mode/)
- Works in Section 3 are linked in the table.
