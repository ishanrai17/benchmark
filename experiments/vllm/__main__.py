import argparse
import json

from models import MAX_TOKENS, MODEL, VLLMModel

from .latency import LATENCY_SIZES
from .suite import EXPERIMENTS, WARMUP_SIZE, run_all
from .throughput_sweep import SWEEP_SIZES


def int_list(s):
    try:
        vals = [int(x) for x in s.replace(" ", "").split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {s!r}")
    if not vals or any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError(f"sizes must be positive ints, got {s!r}")
    return vals


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m experiments.vllm",
        description="vLLM continuous-batching benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=("all",) + EXPERIMENTS,
        help="which experiment to run",
    )

    g = p.add_argument_group("engine")
    g.add_argument("--model", default=MODEL, help="HF model id")
    g.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="tokens per request")
    g.add_argument("--dtype", default="half", help="engine dtype")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--max-model-len", type=int, default=2048)
    g.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="allow CUDA graphs / torch.compile (faster; needs compute capability >= 8)",
    )
    g.add_argument(
        "--warmup-size",
        type=int,
        default=WARMUP_SIZE,
        help="prompts in the warmup batch (0 to skip warmup)",
    )

    g = p.add_argument_group("experiment")
    g.add_argument(
        "--sweep-sizes", type=int_list, default=SWEEP_SIZES, help="e.g. 1,16,64"
    )
    g.add_argument(
        "--latency-sizes", type=int_list, default=LATENCY_SIZES, help="e.g. 1,32"
    )
    g.add_argument("--repeats", type=int, default=3, help="latency runs to average")

    g = p.add_argument_group("output")
    g.add_argument("--outfile", default="benchmark.png", help="plot destination")
    g.add_argument("--no-plot", dest="plot", action="store_false", help="skip plotting")
    g.add_argument(
        "--show", action="store_true", help="open the plot window (off by default on CLI)"
    )
    g.add_argument("--json", metavar="PATH", help="also write raw results as JSON")
    g.add_argument("-q", "--quiet", dest="verbose", action="store_false")

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    model = VLLMModel(
        model=args.model,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    results = run_all(
        model,
        which=args.experiment,
        sweep_sizes=args.sweep_sizes,
        latency_sizes=args.latency_sizes,
        repeats=args.repeats,
        warmup_size=args.warmup_size,
        plot=args.plot,
        outfile=args.outfile,
        show=args.show,
        verbose=args.verbose,
    )

    if args.plot and len(results) < 2 and args.verbose:
        print(f"\n(skipping plot: '{args.experiment}' alone has no sweep+latency pair)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        if args.verbose:
            print(f"wrote {args.json}")

    return results


if __name__ == "__main__":
    main()
