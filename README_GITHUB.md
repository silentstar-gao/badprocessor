# Processor and vLLM inference runners

This repository contains the reusable C-code logits Processors, their FIM
adapters, and the vLLM runners used for attack and probability-monitoring
experiments.

## Included source

- `processors/`: ordinary Transformers logits Processors.
- `processors_fim/`: vLLM/FIM adapters and the shared adapter base.
- `tools/processor_probability_monitor.py`: outer decorator that records
  forced-token probabilities without modifying a Processor.
- `tools/run_vllm_multimodel_attack_monitored.py`: four-model runner at
  temperature `0.4`, with per-request progress, JSONL monitoring, resumable
  terminal records, and automatic shutdown after a successful full run.
- `tools/run_processor_probability_monitor_validation.py`: native vLLM
  validation runner.
- `tools/run_vllm_multimodel_attack.py` and `tools/run_vllm_clean.py`:
  shared attack/clean helpers and the original runners.

Large local assets are intentionally excluded from Git: model weights,
datasets, generated outputs, restored projects, and Python caches.

## Runtime dependencies

Python 3.10+, PyTorch, Transformers, and vLLM 0.4.x are required for model
inference. The runner expects the local experiment layout containing the
dataset and model directories, or explicit `--dataset` and `--models`
arguments.

Compile the included code with:

```bash
python -m py_compile \
  processors/*.py processors_fim/*.py \
  tools/processor_probability_monitor.py \
  tools/run_vllm_multimodel_attack_monitored.py \
  tools/run_processor_probability_monitor_validation.py \
  tools/run_vllm_multimodel_attack.py tools/run_vllm_clean.py
```

Run a safe small test without shutdown:

```bash
python -u tools/run_vllm_multimodel_attack_monitored.py \
  --models /path/to/model \
  --dataset /path/to/trigger_completion_tasks.partial.jsonl \
  --sample-ids SAMPLE_ID_1 SAMPLE_ID_2 SAMPLE_ID_3 \
  --temperature 0.4 --repeats 1 --no-shutdown \
  --output outputs/monitor-smoke
```

The full runner defaults to `max_new_tokens=256`, `top_p=0.95`,
`max_num_seqs=30`, BF16, temperature `0.4`, and 10 repetitions per task.
It writes `results.jsonl`, `skipped.jsonl`,
`processor_probability_events.jsonl`, and `summary.json` per model. Use
`--no-shutdown` during development; the default full-run behavior invokes
`/usr/bin/shutdown` only after all model summaries are written.
