import statistics

from .workload import make_workload

LATENCY_SIZES = [1, 8, 32, 128]


def latency_at(model, n, repeats=3):
    times = [model.time_batch(make_workload(n)) for _ in range(repeats)]
    return statistics.mean(times)


def run_latency(model, sizes=LATENCY_SIZES, repeats=3, verbose=True):
    latency = {}
    for n in sizes:
        wall = latency_at(model, n, repeats)
        per_req_ms = (wall / n) * 1000
        latency[n] = {"wall_s": wall, "per_req_ms": per_req_ms}
        if verbose:
            print(
                f"n={n:3d}   batch wall: {wall:5.2f}s   "
                f"per-request avg: {per_req_ms:6.0f} ms"
            )
    return latency
