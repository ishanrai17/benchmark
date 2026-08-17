from .workload import make_workload

SWEEP_SIZES = [1, 4, 16, 32, 64, 128, 256]


def run_sweep(model, sizes=SWEEP_SIZES, verbose=True):
    sweep = {}
    for n in sizes:
        tps, dt, toks = model.run(make_workload(n))
        sweep[n] = {"tps": tps, "wall_s": dt, "tokens": toks}
        if verbose:
            print(f"n={n:4d}   {tps:7.1f} tok/s   ({toks} toks in {dt:.1f}s)")
    return sweep
