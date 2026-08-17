from .latency import LATENCY_SIZES, run_latency
from .plot_results import plot_results
from .throughput_sweep import SWEEP_SIZES, run_sweep
from .workload import make_workload

WARMUP_SIZE = 2
EXPERIMENTS = ("sweep", "latency")


def run_all(
    model,
    which="all",
    sweep_sizes=None,
    latency_sizes=None,
    repeats=3,
    warmup_size=WARMUP_SIZE,
    plot=True,
    outfile="benchmark.png",
    show=True,
    verbose=True,
):
    if warmup_size > 0:
        model.warmup(make_workload(warmup_size))
        if verbose:
            print("vLLM ready.\n")

    results = {}
    if which in ("all", "sweep"):
        results["sweep"] = run_sweep(
            model, sizes=sweep_sizes or SWEEP_SIZES, verbose=verbose
        )
    if which in ("all", "latency"):
        if verbose and which == "all":
            print()
        results["latency"] = run_latency(
            model,
            sizes=latency_sizes or LATENCY_SIZES,
            repeats=repeats,
            verbose=verbose,
        )

    if plot and "sweep" in results and "latency" in results:
        plot_results(results["sweep"], results["latency"], outfile=outfile, show=show)
    return results
