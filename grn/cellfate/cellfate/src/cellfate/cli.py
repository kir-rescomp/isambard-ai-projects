"""Command-line interface.

    cellfate selftest            correctness: propensities, moments, WE vs brute force
    cellfate bench               throughput and strong/weak scaling measurement
    cellfate run                 one weighted-ensemble run from a YAML config
    cellfate sweep               a grid of runs over cytokine inputs
    cellfate aggregate           collapse a sweep directory into one table

All commands are safe to launch under ``torchrun``; rank 0 handles printing and
writing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__, models
from .dist.comm import make_communicator
from .engine.torch_backend import TorchEngine
from .io.results import aggregate_sweep, provenance, write_run
from .model import propensities_numpy
from .we import BinScheme, WEConfig, WeightedEnsemble
from .we.driver import brute_force_mfpt


# ---------------------------------------------------------------------- #
# helpers


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


def build_model(name: str, params: dict):
    fn = {"toggle": models.toggle, "th17_treg": models.th17_treg, "birth_death": models.birth_death}[name]
    return fn(**params)


def build_pcoord(name: str, net):
    if name == "toggle":
        return models.ratio_pcoord(net, "A", "B")
    if name == "th17_treg":
        return models.ratio_pcoord(net, "RORgt", "FOXP3")
    raise ValueError(f"no default progress coordinate for {name!r}")


def equilibrate(engine, net, model_name: str, n: int, device, scale: float, tau: float, steps: int, seed: int):
    """Build a source-basin reservoir by relaxing a biased initial condition."""
    x = torch.zeros((n, net.n_species), dtype=torch.int64, device=device)
    if model_name == "toggle":
        x[:, net.index("B")] = int(40 * scale)
    else:
        p = getattr(net, "p_max", 400.0 * scale)
        x[:, net.index("FOXP3")] = int(p)
        x[:, net.index("mFOXP3")] = max(1, int(4 * scale))
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x, _ = engine.advance(x, tau, steps, generator=g)
    return x


# ---------------------------------------------------------------------- #
# selftest


def cmd_selftest(args) -> int:
    device = pick_device(args.device)
    print(f"cellfate {__version__} selftest on {device}")
    print(json.dumps(provenance(), indent=2))
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    # 1. propensity agreement against the NumPy definition
    for mname in ("birth_death", "toggle", "th17_treg"):
        net = build_model(mname, {}).compile()
        eng = TorchEngine(net, device=device, dtype=torch.float64)
        x = torch.randint(0, 400, (256, net.n_species), device=device)
        a_t = eng.propensities(x).cpu().numpy()
        a_n = propensities_numpy(net, x.cpu().numpy())
        rel = np.abs(a_t - a_n) / (np.abs(a_n) + 1e-30)
        check(f"propensities/{mname}", rel.max() < 1e-10, f"max rel err {rel.max():.2e}")

    # 2. birth-death stationary moments against Poisson(k/d)
    net = models.birth_death(k=20.0, d=1.0).compile()
    eng = TorchEngine(net, device=device)
    x = torch.zeros((args.walkers, 1), dtype=torch.int64, device=device)
    x, diag = eng.advance(x, 0.02, 1500)
    m, v = float(x.double().mean()), float(x.double().var())
    check("birth_death/mean", abs(m - 20.0) < 0.4, f"{m:.3f} vs 20.000")
    check("birth_death/variance", abs(v - 20.0) < 1.2, f"{v:.3f} vs 20.000")
    check("birth_death/no_clamping", diag.clamped == 0, f"clamped={diag.clamped}")

    # 3. weighted ensemble rate against brute force on a crossable barrier
    scale = 0.5
    net = models.toggle(scale=scale).compile()
    eng = TorchEngine(net, device=device)
    pc = models.ratio_pcoord(net, "A", "B")
    bins = BinScheme.uniform(-0.9, 0.9, 24)
    basin = equilibrate(eng, net, "toggle", args.walkers, device, scale, 0.01, 1000, seed=1)

    bf, hits, n = brute_force_mfpt(eng, bins, pc, basin, 0.01, 200, args.bf_generations, seed=11)
    cfg = WEConfig(
        tau=0.01,
        n_substeps=200,
        n_generations=args.we_generations,
        walkers_per_bin=32,
        initial_walkers=768,
        seed=5,
        log_every=max(1, args.we_generations // 3),
    )
    we = WeightedEnsemble(net, eng, bins, pc, basin[:768], cfg)
    acc = we.run(verbose=args.verbose)
    mfpt, se = acc.mfpt(burn_in=0.3)
    ratio = mfpt / bf if bf > 0 else float("inf")
    check("we/weight_conservation", abs(we.logs[-1].total_weight - 1.0) < 1e-9,
          f"total weight {we.logs[-1].total_weight:.12f}")
    check("we/mfpt_vs_brute_force", 0.6 < ratio < 1.7,
          f"WE {mfpt:.1f}+/-{se:.1f} vs BF {bf:.1f} ({hits}/{n} hits), ratio {ratio:.3f}")
    q = acc.committor(bins.state_a, bins.state_b)
    # A small negative step between adjacent bins is finite-sample noise, not a
    # correctness failure; a material reversal is.
    check("we/committor_monotone", bool(np.all(np.diff(q) >= -0.05)),
          f"largest reversal {min(0.0, float(np.diff(q).min())):.4f}")
    check("we/committor_midpoint", 0.25 < q[len(q) // 2] < 0.75, f"q(mid)={q[len(q)//2]:.3f}")

    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


# ---------------------------------------------------------------------- #
# bench


def cmd_bench(args) -> int:
    device = pick_device(args.device)
    comm = make_communicator(device=device)
    net = build_model(args.model, {"scale": args.scale}).compile()
    eng = TorchEngine(net, device=device, dtype=getattr(torch, args.dtype))

    x = torch.zeros((args.walkers, net.n_species), dtype=torch.int64, device=device)
    x[:, -1] = 100
    eng.advance(x[: min(1024, args.walkers)], args.tau, 5)  # warm up
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    x, diag = eng.advance(x, args.tau, args.steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    local = args.walkers * args.steps / dt
    total = comm.all_reduce_scalar(local)
    if comm.rank == 0:
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(
            f"model={args.model} scale={args.scale} dtype={args.dtype} "
            f"ranks={comm.world_size} walkers/rank={args.walkers} steps={args.steps}"
        )
        print(f"  per rank : {local:.3e} walker-steps/s   ({dt:.2f} s wall)")
        print(f"  aggregate: {total:.3e} walker-steps/s")
        print(f"  peak GPU memory {mem:.2f} GiB, subdivisions {diag.subdivisions}, clamped {diag.clamped}")
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "provenance": provenance(),
                        "model": args.model,
                        "scale": args.scale,
                        "dtype": args.dtype,
                        "ranks": comm.world_size,
                        "walkers_per_rank": args.walkers,
                        "steps": args.steps,
                        "wall_seconds": dt,
                        "walker_steps_per_second_per_rank": local,
                        "walker_steps_per_second_total": total,
                        "peak_gpu_gib": mem,
                    },
                    indent=2,
                )
            )
    comm.shutdown()
    return 0


# ---------------------------------------------------------------------- #
# run


def load_config(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def run_one(conf: dict, device, comm, out_path: Path | None, verbose: bool = True):
    mname = conf["model"]["name"]
    mparams = {k: v for k, v in conf["model"].items() if k != "name"}
    net = build_model(mname, mparams).compile()
    eng = TorchEngine(net, device=device, dtype=getattr(torch, conf.get("dtype", "float64")))
    pc = build_pcoord(mname, net)

    bconf = conf.get("bins", {})
    bins = BinScheme.uniform(
        bconf.get("lo", -0.9), bconf.get("hi", 0.9), bconf.get("n_bins", 32)
    )
    cfg = WEConfig(**conf.get("we", {}))
    basin = equilibrate(
        eng, net, mname, conf.get("basin_walkers", 2048), device,
        mparams.get("scale", 1.0), cfg.tau, conf.get("basin_steps", 2000), seed=cfg.seed,
    )
    we = WeightedEnsemble(net, eng, bins, pc, basin, cfg, comm=comm)
    acc = we.run(verbose=verbose and comm.rank == 0)
    summary = acc.summary(bins.state_a, bins.state_b)
    if comm.rank == 0 and out_path is not None:
        write_run(out_path, summary, we.logs, conf.get("we", {}), mparams, acc.transitions)
    return summary


def cmd_run(args) -> int:
    device = pick_device(args.device)
    comm = make_communicator(device=device)
    conf = load_config(args.config)
    if args.override:
        for kv in args.override:
            key, val = kv.split("=", 1)
            node = conf
            *path, last = key.split(".")
            for p in path:
                node = node.setdefault(p, {})
            node[last] = yaml_scalar(val)
    out = Path(args.out) if args.out else None
    summary = run_one(conf, device, comm, out, verbose=not args.quiet)
    if comm.rank == 0:
        print(
            f"\nMFPT {summary['mfpt']:.4e} +/- {summary['mfpt_stderr']:.2e}   "
            f"flux {summary['flux']:.4e}   generations {summary['generations']}"
        )
        if out:
            print(f"written: {out}")
    comm.shutdown()
    return 0


def yaml_scalar(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


# ---------------------------------------------------------------------- #
# sweep


def cmd_sweep(args) -> int:
    """A grid of independent weighted-ensemble runs.

    Each grid point is one run.  With ``--shard-by-rank`` the grid is divided
    across ranks and each rank runs its points serially with no communication,
    which is the right layout when the grid is larger than the node count.
    Without it, every rank cooperates on each grid point in turn, which is the
    right layout when a single point needs more walkers than one GPU holds.
    """
    device = pick_device(args.device)
    comm = make_communicator(distributed=not args.shard_by_rank, device=device)
    conf = load_config(args.config)
    sweep = conf.get("sweep", {})
    keys = list(sweep.keys())
    grid = list(itertools.product(*[sweep[k] for k in keys]))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0")) if args.shard_by_rank else comm.rank
    world = int(os.environ.get("WORLD_SIZE", "1")) if args.shard_by_rank else comm.world_size
    mine = [(i, p) for i, p in enumerate(grid) if not args.shard_by_rank or i % world == rank]

    for i, point in mine:
        c = json.loads(json.dumps(conf))
        for k, v in zip(keys, point):
            c["model"][k] = v
        tag = "_".join(f"{k}{v}" for k, v in zip(keys, point)).replace(".", "p")
        out = outdir / f"run_{i:05d}_{tag}.json"
        if out.exists() and not args.force:
            continue
        t0 = time.perf_counter()
        s = run_one(c, device, comm if not args.shard_by_rank else make_communicator(False), out, verbose=False)
        print(
            f"[rank {rank}] {i+1}/{len(grid)} {tag}: MFPT {s['mfpt']:.4e} "
            f"({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )
    if not args.shard_by_rank:
        comm.shutdown()
    return 0


def cmd_aggregate(args) -> int:
    p = aggregate_sweep(args.dir, args.out)
    print(f"written: {p}")
    return 0


# ---------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cellfate", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(device=("--device", "auto"))

    p = sub.add_parser("selftest", help="correctness checks")
    p.add_argument("--device", default="auto")
    p.add_argument("--walkers", type=int, default=3000)
    p.add_argument("--we-generations", type=int, default=1200)
    p.add_argument("--bf-generations", type=int, default=1200)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("bench", help="throughput measurement")
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default="th17_treg")
    p.add_argument("--scale", type=float, default=3.0)
    p.add_argument("--walkers", type=int, default=100_000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("run", help="one weighted-ensemble run")
    p.add_argument("config")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--override", action="append", default=[], metavar="a.b=value")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="grid of runs over model parameters")
    p.add_argument("config")
    p.add_argument("--outdir", default="results")
    p.add_argument("--device", default="auto")
    p.add_argument("--shard-by-rank", action="store_true",
                   help="one grid point per rank (no collectives) instead of all ranks per point")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("aggregate", help="collapse a sweep directory into one table")
    p.add_argument("dir")
    p.add_argument("--out", default="sweep_summary")
    p.set_defaults(func=cmd_aggregate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
