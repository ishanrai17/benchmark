def plot_results(sweep, latency, outfile="benchmark.png", show=True):

    import matplotlib.pyplot as plt

    xs = sorted(sweep.keys())
    tps_vals = [sweep[n]["tps"] for n in xs]
    baseline = sweep[xs[0]]["tps"]
    speedups = [t / baseline for t in tps_vals]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(xs, tps_vals, "o-", color="#27ae60", linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Throughput vs concurrency")
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(xs, speedups, "o-", color="#2980b9", linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel(f"Speedup vs sequential (n={xs[0]} = {baseline:.0f} tok/s)")
    ax.set_title("Continuous batching speedup")
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    lx = sorted(latency.keys())
    ly = [latency[n]["per_req_ms"] for n in lx]
    ax.plot(lx, ly, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Avg per-request latency (ms)")
    ax.set_title("Latency cost of higher concurrency")
    ax.set_xticks(lx)
    ax.set_xticklabels(lx)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches="tight")
    if show:
        plt.show()

    peak_n = max(sweep, key=lambda n: sweep[n]["tps"])
    peak = sweep[peak_n]["tps"]
    print(f"\nSequential baseline (n={xs[0]}):  {baseline:6.0f} tok/s")
    print(
        f"Peak throughput   (n={peak_n:3d}): {peak:6.0f} tok/s   "
        f"({peak / baseline:4.1f}x speedup)"
    )
    return baseline, peak_n, peak
