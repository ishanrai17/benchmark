# vLLM continuous-batching benchmarks

Measures how throughput and per-request latency scale with concurrency on a vLLM engine.

- `models/` — the engine wrapper (`VLLMModel`)
- `experiments/vllm/` — the workload, the experiments, and the CLI

## Setup

```bash
pip install -r requirements.txt
```

Needs a GPU. On Colab: `!pip install -q vllm matplotlib`

## Running

Run from the **repo root** — imports like `from models import ...` are resolved relative
to it, and there is no `pyproject.toml` to install the packages onto your path.

```bash
python -m experiments.vllm                              # sweep + latency + plot
python -m experiments.vllm sweep --sweep-sizes 1,8,64
python -m experiments.vllm latency --repeats 5
python -m experiments.vllm --json results.json --no-plot
```

`python -m experiments.vllm --help` lists every flag: engine config (`--model`,
`--dtype`, `--max-tokens`, `--no-enforce-eager`, ...), experiment config
(`--sweep-sizes`, `--latency-sizes`, `--repeats`, `--warmup-size`), and output
(`--outfile`, `--no-plot`, `--show`, `--json`, `-q`).

The plot needs both experiments, so it is skipped when you run only `sweep` or `latency`.

On Colab:

```python
!cd /content/benchmarks && python -m experiments.vllm
```

To match the original notebook's in-process engine and quieter logs, set this before
running — the engine otherwise starts in a subprocess:

```python
%env VLLM_ENABLE_V1_MULTIPROCESSING=0
```

## From code

```python
from models import VLLMModel
from experiments.vllm import run_all

results = run_all(VLLMModel())      # {"sweep": {...}, "latency": {...}}
```

`run_all` takes `which="sweep"|"latency"|"all"`, `sweep_sizes`, `latency_sizes`,
`repeats`, `warmup_size` (0 skips warmup), `plot`, `outfile`, `show`, and `verbose`.
The individual pieces — `run_sweep`, `run_latency`, `plot_results`, `make_workload`,
`PROMPTS` — are importable from `experiments.vllm` too.
